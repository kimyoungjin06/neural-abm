# Rank 3 production disposition

Medicine rank 3, T10543, entry 1986 is frozen as `CONTENT_FAIL` under
candidate rule v2. Its recent window did not cross the majority-opaque branch,
but it did cross the retained joint anomaly branch:

- median topic score fell from `0.97865` to `0.04635`;
- the score drop was `0.93230`;
- dominant-source share rose from `0.31250` to `0.84375`;
- the concentration increase was `0.53125`.

This adds the sixth production disposition and creates frontier v4. Medicine
ranks 1, 2, and 3 now fail the indexing-quality screen; rank 6 remains a frozen
pass. The lazy frontier therefore requires rank 4, T12505, entry 1993. If rank
4 passes, the compatible Medicine pair is `(4, 6)`; otherwise the frontier
must expand to rank 5.

No human or API label was used as ground truth. No Medicine or global
selection certificate, deterministic reselection, outcome read, model fit, or
scientific-effect claim is valid at this state.
