# Section 10 Draft: Conclusion

Status: draft prose candidate.

Neural ABM separates policy learning from social interaction. A fixed rule,
a table written before the run, a policy that keeps learning from the run's
own outcomes, and an external prior elicited once from a language model all
fill the same policy slot. Each nests its predecessor exactly, seeds are
paired, and reruns are byte identical on one host. The choice between them
thus becomes a modeling decision that can be tested, rather than one made
implicitly by picking a framework.

In the synthetic designs reported here, that test gave conditional answers.
A table written before the run sufficed in stationary worlds. The adaptive
policy won once about a quarter of runs carried information that the donor
runs lacked. A language model consulted once took under 10 GPU minutes,
where querying it at every agent-step of the same design was estimated at 38
to 125 GPU hours. Inside the learner, though, it lowered the agents' utility.
The
lifecycle could trace that loss to the prior's parts and correct the trace
when its calibration was matched in the world rather than on a design
cohort.

None of these answers is the claim. The claim is that the lifecycle makes
such answers possible to obtain, check, and narrow. The primitive should be
earned through ablations like these, not assumed. Whether neural policies
are necessary, and whether the conditional answers hold in real data, remain
open.
