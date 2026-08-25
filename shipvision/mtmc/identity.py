"""Stable cross-camera identity: turning this instant's clusters into lasting global ids.

The clusterer answers "which of the tracks visible right now are the same object". This
answers the harder question: "and is that object the one we called 41 a minute ago". It is the
only stateful component in the package, it is where every hard case lives, and it is the part
of the reference implementation that had to be studied rather than translated.

What is ported, because it is genuinely good:

* **Largest clusters first, ties by first appearance.** A cluster of four views is much more
  likely to be the identity that owns a global id than a cluster of one, so it gets first
  claim on it. Processing in arbitrary order lets a single spurious view steal an id from the
  group that has been carrying it.
* **All maximum-intersection candidates, not the first.** When a cluster overlaps several
  known identities equally, the reference returns every one of them and then chooses. An
  earlier version returned the first match found, which made the assignment depend on hash
  order — the same footage, replayed, produced different ids.
* **Arbitrate a contested id by age, then by appearance.** Among equal candidates, the one
  whose longest-lived member is oldest wins, because it has the most evidence behind it. When
  two tracks from *one* camera both want a place in one identity, the winner is whichever
  looks more like the rest of that identity right now.
* **An invariant check, and something that throws.** ``_owner`` and ``_members`` are two views
  of one fact, and a bug that desynchronises them produces output that is wrong but plausible
  for hours. :meth:`GlobalIdAssigner.validate` states the invariant and raises.

Two things are deliberately **not** ported, because they are bugs.

**Eviction is not a variant.** The reference has a "cleanable" tracker whose whole job is to
prune, and a name-to-id table that maps ``"cleanable_aic_tracker"`` to the plain
``aic_tracker`` — so the shipped configuration builds the version that never prunes, and its
global-id storage and reverse map grow for as long as the process lives. There is no
non-evicting variant here. A capacity and a maximum age are constructor arguments with
defaults, eviction runs on every call including calls with no tracks at all, and
:meth:`sizes` exists so a test can assert on the actual lengths.

**The id counter belongs to the assigner.** The reference's is ``static inline``: one counter
shared by every tracker instance in the process, and ``reset()`` on any one of them sets it
back to zero. Two camera groups in one server therefore interleave their id spaces, and after
a reset the next new identity is issued id 0 — which is still live, still in a database, and
still on somebody's screen. Here the counter is per-instance, monotonic, and
:meth:`reset` does not rewind it: forgetting who an identity was is not the same as being
allowed to reuse its name.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from shipvision.errors import ConfigurationError, TrackingError
from shipvision.mtmc.frames import TrackKey, TrackObservation
from shipvision.reid.distance import cosine_similarity, normalize

__all__ = ["GlobalIdAssigner"]


class GlobalIdAssigner:
    """Assigns and maintains cross-camera global ids across calls.

    One instance per camera group. Not thread-safe on its own: the tracker that owns it holds
    a lock, which is the level where the whole update is one atomic step.
    """

    def __init__(
        self,
        *,
        max_age: int = 30,
        capacity: int = 4096,
        max_tracks: int = 8192,
        validate_every_step: bool = False,
    ) -> None:
        """
        Args:
            max_age: how many consecutive synchronised instants a track may go unobserved
                before it is forgotten. Counted in instants rather than seconds because the
                instant *is* this component's clock: it is called once per synchronised group,
                and wall-clock ageing would make eviction depend on decoder timestamps that
                stop advancing exactly when a camera is in trouble.
            capacity: the maximum number of live global ids. On overflow the least recently
                seen identity is evicted whole. This is the backstop for the case a
                per-track age cannot cover — an incident that produces thousands of
                simultaneous identities — and CLAUDE.md's rule that nothing here grows
                without bound means having it rather than trusting traffic.
            max_tracks: the maximum number of single-camera tracks held across all identities.
                The second bound: one identity that keeps acquiring members must not be able
                to fill memory on its own.
            validate_every_step: run :meth:`validate` after every call. Off by default because
                it is O(tracked) work in service of catching a bug that should not exist; on
                in tests, and worth turning on in a canary deployment.
        """
        if max_age < 1:
            raise ConfigurationError(
                f"max_age must be at least 1 instant; 0 would forget every track between "
                f"consecutive frames, got {max_age}"
            )
        if capacity < 1:
            raise ConfigurationError(f"capacity must be positive, got {capacity}")
        if max_tracks < 1:
            raise ConfigurationError(f"max_tracks must be positive, got {max_tracks}")

        self.max_age = int(max_age)
        self.capacity = int(capacity)
        self.max_tracks = int(max_tracks)
        self.validate_every_step = bool(validate_every_step)

        self._counter = 0
        self._step = 0
        self._members: dict[int, list[TrackKey]] = {}
        self._owner: dict[TrackKey, int] = {}
        self._features: dict[TrackKey, np.ndarray] = {}
        self._hits: dict[TrackKey, int] = {}
        self._last_seen: dict[TrackKey, int] = {}

    # -- introspection ----------------------------------------------------------------

    @property
    def step(self) -> int:
        """How many synchronised instants have been processed. The TTL clock."""
        return self._step

    @property
    def issued(self) -> int:
        """How many global ids have ever been issued. Monotonic, including across resets."""
        return self._counter

    @property
    def global_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._members))

    def members(self, global_id: int) -> tuple[TrackKey, ...]:
        """Every ``(camera_id, track_id)`` currently believed to be this identity."""
        return tuple(self._members.get(global_id, ()))

    def owner_of(self, key: TrackKey) -> int | None:
        """The global id this track belongs to, or `None` if it is not known.

        `None`, never ``-1``. ``-1`` sorts and serialises as an ordinary id, which is how the
        reference's unassigned tracks flowed downstream looking assigned.
        """
        return self._owner.get(key)

    def __len__(self) -> int:
        """How many live global ids."""
        return len(self._members)

    def sizes(self) -> dict[str, int]:
        """Every internal container's length. What a growth test asserts on.

        Exposed as data rather than described in a comment because "bounded" is a claim, and
        the only way to keep a claim true over a codebase's life is to let a test read the
        actual numbers.
        """
        return {
            "global_ids": len(self._members),
            "owner": len(self._owner),
            "features": len(self._features),
            "hits": len(self._hits),
            "last_seen": len(self._last_seen),
            "members_total": sum(len(v) for v in self._members.values()),
        }

    # -- the main entry point ---------------------------------------------------------

    def assign(
        self,
        observations: Sequence[TrackObservation],
        labels: Sequence[int] | np.ndarray,
    ) -> dict[TrackKey, int]:
        """Advance one instant: give every observation a global id.

        Args:
            observations: the tracks that survived gating, in any order.
            labels: one cluster label per observation, as the clusterer produced them. Only
                equality between labels is read.

        Returns:
            Every observation's key mapped to its global id. Never partial and never `None`:
            an observation that reached this point is a track the caller decided to trust, so
            if it matches nothing it starts a new identity rather than being dropped. The
            reference left such tracks at ``-1`` and published them.

        Raises:
            TrackingError: an observation has no embedding, or the internal maps have
                desynchronised.
            ConfigurationError: ``labels`` does not line up with ``observations``.
        """
        if len(labels) != len(observations):
            raise ConfigurationError(
                f"{len(labels)} labels for {len(observations)} observations; these come from "
                f"one clustering call and must line up"
            )

        self._step += 1
        self._observe(observations)

        result: dict[TrackKey, int] = {}
        matched: list[int] = []
        for indices in self._ordered_groups(labels):
            self._assign_group([observations[i] for i in indices], matched)

        for observation in observations:
            owner = self._owner.get(observation.key)
            if owner is None:
                # Unreachable through the branches above; kept because "every observation
                # leaves with an id" is a guarantee this function makes to its caller, and a
                # guarantee that depends on a case analysis being exhaustive should be
                # enforced rather than believed.
                owner = self._issue()
                self._adopt(observation.key, owner)
            result[observation.key] = owner

        self._evict()
        if self.validate_every_step:
            self.validate()
        return result

    def reset(self) -> None:
        """Forget every identity. Does **not** rewind the id space.

        The next identity discovered gets the next number, not zero. Ids that have been
        published are out in the world — in a database, on an operator's screen, in a
        downstream tracker — and reusing one attaches a stranger to that history.
        """
        self._members.clear()
        self._owner.clear()
        self._features.clear()
        self._hits.clear()
        self._last_seen.clear()

    def validate(self) -> None:
        """Assert that the maps agree, or raise.

        ``_owner`` and ``_members`` are the same fact stored twice, for the same reason a
        database has an index: one direction is needed per cluster and the other per track.
        Two representations of one fact can disagree, and when they do the output stays
        plausible — an identity that quietly holds two tracks from one camera, an id that
        nothing can be found under — so the invariant is written down and checked.
        """
        for global_id, keys in self._members.items():
            if not keys:
                raise TrackingError(
                    f"global id {global_id} has no members; an empty identity should have "
                    f"been dropped when its last member left"
                )
            if len(set(keys)) != len(keys):
                raise TrackingError(f"global id {global_id} lists a member twice: {keys}")
            cameras = [key.camera_id for key in keys]
            if len(set(cameras)) != len(cameras):
                raise TrackingError(
                    f"global id {global_id} holds two tracks from one camera: {keys}. It is "
                    f"one object; a camera seeing it twice at one instant is a "
                    f"single-camera tracking failure, and every per-camera lookup here "
                    f"would resolve to whichever of the two came first"
                )
            for key in keys:
                owner = self._owner.get(key)
                if owner != global_id:
                    raise TrackingError(
                        f"{key} is a member of global id {global_id} but its owner is "
                        f"{owner}; the forward and reverse maps have desynchronised"
                    )
        for key, global_id in self._owner.items():
            if key not in self._members.get(global_id, ()):
                raise TrackingError(
                    f"{key} claims global id {global_id}, which does not list it"
                )
            if key not in self._features or key not in self._last_seen:
                raise TrackingError(
                    f"{key} is owned but has no stored feature or last-seen instant, so it "
                    f"can never be arbitrated or evicted"
                )
        extra = set(self._features) - set(self._owner)
        if extra:
            raise TrackingError(
                f"{len(extra)} tracks have stored features but no owner (first: "
                f"{sorted(extra)[0]}); these would never be evicted"
            )

    # -- per-instant bookkeeping ------------------------------------------------------

    def _observe(self, observations: Sequence[TrackObservation]) -> None:
        """Record this instant's features, consecutive-hit counts and last-seen stamps."""
        for observation in observations:
            embedding = observation.embedding
            if embedding is None:
                raise TrackingError(
                    f"{observation.key} has no embedding; cross-camera identity is decided "
                    f"on appearance, so an un-embedded track cannot be assigned"
                )
            key = observation.key
            self._features[key] = normalize(np.asarray(embedding, dtype=np.float32))
            previous = self._last_seen.get(key)
            self._hits[key] = self._hits.get(key, 0) + 1 if previous == self._step - 1 else 1
            self._last_seen[key] = self._step

    @staticmethod
    def _ordered_groups(labels: Sequence[int] | np.ndarray) -> list[list[int]]:
        """Indices grouped by label, largest group first, ties by first appearance.

        The reference's ``reorderCluster``. Deterministic ordering is the point: with ties
        broken by first appearance rather than by dictionary order, the same input produces
        the same ids twice, which is the difference between a reproducible bug and a haunting.
        """
        values = [int(label) for label in labels]
        counts = Counter(values)
        first: dict[int, int] = {}
        for index, value in enumerate(values):
            first.setdefault(value, index)
        order = sorted(counts, key=lambda value: (-counts[value], first[value]))
        grouped: dict[int, list[int]] = {value: [] for value in order}
        for index, value in enumerate(values):
            grouped[value].append(index)
        return [grouped[value] for value in order]

    # -- the assignment itself --------------------------------------------------------

    def _assign_group(self, group: Sequence[TrackObservation], matched: list[int]) -> None:
        """Give one cluster of observations a global id, resolving contested claims."""
        keys = [observation.key for observation in group]
        candidates = self._candidate_ids(keys, matched)

        if not candidates:
            fresh = [key for key in keys if key not in self._owner]
            if not fresh:
                # Every member already has an id and no id wants this cluster: two known
                # identities that this instant thinks are one. Splitting or merging them on
                # one frame's evidence is how identities oscillate, so nothing happens.
                return
            # One identity per camera, even here. Two same-camera keys can only reach one
            # cluster if the matcher's exclusion mask was bypassed, and the reference
            # trusted that it never would be — so its brand-new identities could contain two
            # tracks from one camera, which nothing downstream expects and its own
            # per-camera lookup then resolves arbitrarily. Enforcing it locally means this
            # class holds its invariant whatever it is handed.
            global_id = self._issue()
            claimed: set[str] = set()
            for key in fresh:
                if key.camera_id in claimed:
                    self._adopt(key, self._issue())
                    continue
                claimed.add(key.camera_id)
                self._adopt(key, global_id)
            matched.append(global_id)
            return

        target = self._select_by_oldest(candidates)
        matched.append(target)
        members = self._members[target]
        overlap = [key for key in keys if key in members]
        non_overlap = [key for key in keys if key not in members]
        overlap_features = [self._features[key] for key in overlap]
        displaced: list[TrackKey] = []

        for key in non_overlap:
            incumbent = self._member_from_camera(target, key.camera_id)
            owner = self._owner.get(key)

            if incumbent is None:
                if owner is None:
                    self._adopt(key, target)
                elif owner != target:
                    self._resolve_between_identities(key, owner, target, overlap_features)
                continue

            if self._last_seen.get(incumbent, -1) < self._step:
                # The camera's slot in this identity is held by a track nobody has seen this
                # instant. A live track outranks a stale one: the single-camera tracker
                # replaced it, which is exactly what happens when a person is briefly
                # occluded and comes back with a new track id.
                self._forget(incumbent)
                self._place(key, target, owner)
                continue

            contest = self._similarity(self._features[incumbent], overlap_features, "mean")
            challenge = self._similarity(self._features[key], overlap_features, "mean")
            if challenge > contest:
                self._place(key, target, owner)
                displaced.append(incumbent)
            elif owner is None:
                # It lost the contest and has no identity of its own: it is a real object
                # this identity has no room for, so it becomes its own identity rather than
                # being published unassigned.
                self._adopt(key, self._issue())

        # Losers are moved out only after the whole cluster is processed, so that a chain of
        # contests within one cluster is judged against the state it started from.
        for key in displaced:
            if self._owner.get(key) == target:
                self._move(key, target, self._issue())

    def _resolve_between_identities(
        self,
        key: TrackKey,
        owner: int,
        target: int,
        overlap_features: Sequence[np.ndarray],
    ) -> None:
        """Decide whether an already-identified track should defect to ``target``.

        Its current identity has no track from this camera, so nothing is displaced either
        way; the question is only which group this track looks more like. Compared on the
        *maximum* similarity rather than the mean, because an identity's members are views
        from different angles and the mean punishes an identity for having a bad angle in it,
        which is the normal state of a large group.
        """
        rest = [
            self._features[member]
            for member in self._members.get(owner, ())
            if member != key and member in self._features
        ]
        to_target = self._similarity(self._features[key], overlap_features, "max")
        to_own = self._similarity(self._features[key], rest, "max")
        if to_target > to_own:
            self._move(key, owner, target)

    def _candidate_ids(self, keys: Sequence[TrackKey], exclude: Sequence[int]) -> list[int]:
        """Every global id sharing the most members with this cluster, excluding ``exclude``.

        The reference scans its whole global storage and intersects each identity's member
        list with the cluster. The result is identical to counting how many of the cluster's
        keys each id already owns — because ``_owner[k] == g`` if and only if
        ``k in _members[g]``, which is the invariant :meth:`validate` enforces — and counting
        is O(cluster) rather than O(live identities). At a capacity of 4 096 identities and
        fifty clusters an instant, that is the difference between a scan of two hundred
        thousand member lists per instant and fifty dictionary lookups.

        Returns them sorted, so a tie downstream resolves the same way every time.
        """
        blocked = set(exclude)
        counts = Counter(
            owner
            for owner in (self._owner.get(key) for key in keys)
            if owner is not None and owner not in blocked
        )
        if not counts:
            return []
        best = max(counts.values())
        return sorted(global_id for global_id, count in counts.items() if count == best)

    def _select_by_oldest(self, candidates: Sequence[int]) -> int:
        """Of several equally-overlapping identities, the one with the oldest member.

        Age here is consecutive observations, so "oldest" means "has been continuously
        confirmed the longest" rather than "was created first" — an identity that has been
        watched for two hundred frames has more evidence behind it than one created forty
        minutes ago and seen twice. Ties go to the lowest id, which by construction is the
        one created first.
        """
        if len(candidates) == 1:
            return candidates[0]
        best_id, best_age = candidates[0], -1
        for global_id in candidates:
            age = max(
                (self._hits.get(member, 0) for member in self._members.get(global_id, ())),
                default=0,
            )
            if age > best_age:
                best_id, best_age = global_id, age
        return best_id

    def _member_from_camera(self, global_id: int, camera_id: str) -> TrackKey | None:
        """This identity's existing track on ``camera_id``, if it has one.

        One identity may hold at most one track per camera: it is one object, and a camera
        that sees one object twice at one instant has a single-camera tracking failure, not a
        cross-camera one. That constraint is what makes the contest below necessary.
        """
        for member in self._members.get(global_id, ()):
            if member.camera_id == camera_id:
                return member
        return None

    @staticmethod
    def _similarity(feature: np.ndarray, others: Sequence[np.ndarray], mode: str) -> float:
        """Cosine similarity of one embedding against several, reduced by mean or max.

        Zero when there is nothing to compare against — not an error and not one. A cluster
        whose overlap with an identity is empty gives every contender the same score, which
        correctly makes appearance silent and lets the caller's other rules decide.
        """
        if not others:
            return 0.0
        scores = cosine_similarity(feature[None, :], np.stack(others))[0]
        return float(np.mean(scores)) if mode == "mean" else float(np.max(scores))

    # -- storage mutation -------------------------------------------------------------

    def _issue(self) -> int:
        """The next global id. Monotonic for the life of the instance."""
        global_id = self._counter
        self._counter += 1
        return global_id

    def _adopt(self, key: TrackKey, global_id: int) -> None:
        self._owner[key] = global_id
        self._members.setdefault(global_id, []).append(key)

    def _move(self, key: TrackKey, source: int, destination: int) -> None:
        if source == destination:
            return
        self._detach(key, source)
        self._adopt(key, destination)

    def _place(self, key: TrackKey, target: int, owner: int | None) -> None:
        """Adopt or move, depending on whether the track already had an identity."""
        if owner is None:
            self._adopt(key, target)
        else:
            self._move(key, owner, target)

    def _detach(self, key: TrackKey, global_id: int) -> None:
        members = self._members.get(global_id)
        if members is None:
            return
        if key in members:
            members.remove(key)
        if not members:
            # Dropped as soon as it empties. The reference kept recently-created empty
            # identities around on the theory that they might come back; nothing ever looked
            # them up, so they were a slow leak with a comment on it.
            del self._members[global_id]

    def _forget(self, key: TrackKey) -> None:
        """Remove one track from every map. The only way state leaves this class."""
        owner = self._owner.pop(key, None)
        if owner is not None:
            self._detach(key, owner)
        self._features.pop(key, None)
        self._hits.pop(key, None)
        self._last_seen.pop(key, None)

    def _forget_all(self, keys: Iterable[TrackKey]) -> None:
        for key in keys:
            self._forget(key)

    def _evict(self) -> None:
        """Age out stale tracks, then enforce both capacities. Runs on every instant.

        Including instants with no tracks at all, which is the case that matters: a camera
        group that goes quiet is exactly when a non-evicting implementation stops evicting.
        """
        cutoff = self._step - self.max_age
        self._forget_all([key for key, seen in self._last_seen.items() if seen < cutoff])

        if len(self._owner) > self.max_tracks:
            oldest = sorted(self._last_seen, key=lambda key: (self._last_seen[key], key))
            self._forget_all(oldest[: len(self._owner) - self.max_tracks])

        if len(self._members) > self.capacity:
            # An identity's recency is that of its most recently seen member: an identity
            # with one live track is in use, however old its other members are. Ties break
            # towards evicting the *higher* id, which is the more recently created identity
            # and so the one with the least history behind it — and, more importantly, it
            # breaks them the same way every run.
            def recency(global_id: int) -> tuple[int, int]:
                members = self._members[global_id]
                return (
                    max((self._last_seen.get(m, -1) for m in members), default=-1),
                    -global_id,
                )

            stale = sorted(self._members, key=recency)[: len(self._members) - self.capacity]
            for global_id in stale:
                self._forget_all(list(self._members.get(global_id, ())))

    def __repr__(self) -> str:
        return (
            f"<GlobalIdAssigner identities={len(self._members)} tracks={len(self._owner)} "
            f"issued={self._counter} step={self._step} max_age={self.max_age}>"
        )
