# Setup Guide
(primarily tested on MacOS)
This setup guide helps you to reproduce the results from the paper.
The Julia part is a direct copy from the Setup Guide of the factor-graph framework's README.
Use a command line to `cd` into the repository.

## Python Setup
Check if you have Python on your system, we recommend version 3.9.
We advise you to use a activate a virtual environment for the dependencies with pipenv.
You can install pipenv by running
``` bash
pip install pipenv
```
followed by
``` bash
python -m pipenv --python 3.9
```
to create the virtual env.
If it is not activated yet, please run
``` bash
python -m pipenv shell
```
(On some systems, you must type `python3` instead of `python`.)
Then, run
``` bash
pip install -r requirements.txt
```

Run
``` bash
where python
```
to find out the path of the Python installation on your system.

At the beginning of the Julia files, the Python path must be entered.
Copy it there.
In the following line, the path to your repository's local path is needed.
Copy it there.

## Setup Julia
It is useful to create a new user-wide Julia env that is not constrained to one repository or directory. Julia offers this feature as "shared env", which can be used when activating julia in the command line. VSCode also has some button (at the bottom) that allows to set the julia env.


You can create a new Julia env called "myenv" like this:
``` bash
julia --project=@bayesian_actor_critic --threads 1
```

Next, install all dependencies:
``` Julia
import Pkg; Pkg.add(["Adapt", "BenchmarkTools", "CalibrationErrors", "Distributions", "GraphRecipes", "Graphs", "HDF5", "Integrals", "InvertedIndices", "IrrationalConstants", "KernelAbstractions", "MLDatasets", "NNlib", "Plots", "Polyester", "ProgressBars", "QuadGK", "SpecialFunctions", "StatsBase", "Tullio", "Serialization", "PyCall", "Distributed", "Statistics", "ProgressMeter", "LinearAlgebra"])
```

## Running the pipeline
In your Julia environment, run
``` Julia
include("tools/train_agent_single.jl")
```
which should start the training.
This should take around 10 minutes, depending on the machine.
Once the training is finished, you can run
``` Julia
include("tools/run_agent.jl")
```
and check if you see the pendulum.

Now, you are ready to run the full evaluation pipeline with
``` Julia
include("main_evaluation_pipeline.jl")
```
