# Engineering repairs (no method changes)

1. Initial AST extraction compiled every official generator function; an unused
   function default referenced module-global data_dir, causing NameError before
   generation. Extract only build_atomic, build_nhop_facts_fast and split_facts.
   Rerun generation succeeded; no partial generated data was used for training.
2. A batch-file's contents were passed as one SSH command and exited without
   launching the driver. Invoke the uploaded batch file with cmd /c instead.
   No optimizer updates occurred in the unsuccessful launch.
3. The auxiliary post-training text/cache audit checked target column4 as the
   final entity for random OOD records, where column4 is e5 rather than eD.
   Correct the audit to use the absorbing column23. This checker error did not
   affect training or ID evaluation: both use D<=5 and K5 by protocol. OOD sweep
   uses each sample's actual final entity, and the generator stores all24 states.

The methods, architecture, initialization, data sampling and optimization are
unchanged by these engineering repairs. The older project's native runtime
crashes are not reused as evidence about the current experiment.
