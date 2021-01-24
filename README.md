# Sigmoidal NMFD

Code for the paper _Vande Veire, Len and De Bie, Tijl and De Boom, Cedric, "Sigmoidal NMFD: Convolutional NMF with Saturating Activations For Drum Loop Decomposition"_.

## Installation

```bash
# Download the repository
git clone https://github.com/aida-ugent/sigmoidal-nmfd
cd sigmoidal-nmfd

# Install requirements...
# ... for conda users:
conda create --name sigmoidnmfd --file requirements.txt

# ... alternatively:
pip install -r requirements.txt
```

You will also need to install soundfile:

```
sudo apt-get install libsndfile1
```

## Running sigmoidal NMFD 

The script `run_nmfd_sigmoid.py` applies sigmoidal NMFD to the provided audio file.
For example:

```bash
python -m scripts.run_nmfd_sigmoid resources/moonkits-hiphop.wav 4 --plot
```

The original NMFD algorithm can be run as follows:

```bash
python -m scripts.run_nmfd_vanilla resources/moonkits-hiphop.wav 4 --plot
```

The sparse NMFD baseline can be run as follows:

```bash
python -m scripts.run_nmfd_sparsity resources/moonkits-hiphop.wav 4 --sparsity 0.1 --plot
```

## Running the experiments from the paper

First, download the [ENST dataset](http://www.tsi.telecom-paristech.fr/aao/en/2010/02/19/enst-drums-an-extensive-audio-visual-database-for-drum-signals-processing/).

Then, execute the `experiment_nmfdsigmoid_on_enst.py` script:

```bash
python -m scripts.experiment_nmfdsigmoid_on_enst --dir-enst /path/to/ENST-drums-public --dir-out /home/user/somedirectory --tracklist "resources/tracklists/tracklist_enst_allphrases.csv"
```

This will automatically crop the ENST phrase files as described in the paper, save them in a new directory,
and apply all baselines and the proposed sigmoidal model to all cropped phrases in the dataset.
The results are saved in .npz archives (note: this requires about 1 GB of disk space).
It will then print out the metric values aggregated over all examples.

The ablation experiments can be executed analogously.  
For the ablation experiments on sigmoidal NMFD, including the evaluation of the different optimization strategies:

```bash
python -m scripts.experiment_ablation_nmfdsigmoid_on_enst --dir-enst /path/to/ENST-drums-public --dir-out /home/user/somedirectory --tracklist "resources/tracklists/tracklist_enst_allphrases.csv"
```

For sparse NMFD with an unconstrained warm-up stage:

```bash
python -m scripts.experiment_nmfdsparse_with_warmup.py --dir-enst /path/to/ENST-drums-public --dir-out /home/user/somedirectory --tracklist "resources/tracklists/tracklist_enst_allphrases.csv"
```

Note that parallel processing is supported by adding a `--parallel` flag in the aforementioned commands for the ablation experiments.

## Recreating the initialization templates

The initialization values for the templates `W` in the NMFD framework can be recreated using the 
`create_nmf_drum_templates_from_sample_library.py` script.

For example, use [these drum samples](https://www.producerspot.com/download-free-edm-drums-drum-samples-kit-by-producerspot) from Producerspot, as we did for this paper.

Then execute:

```bash
python create_nmf_drum_templates_from_sample_library.py --samples-list-file resources/tracklist/templates/samples_kick.csv --output-file resources/templates/kick.npy
python create_nmf_drum_templates_from_sample_library.py --samples-list-file resources/tracklist/templates/samples_snare.csv --output-file resources/templates/snare.npy
python create_nmf_drum_templates_from_sample_library.py --samples-list-file resources/tracklist/templates/samples_hihat.csv --output-file resources/templates/hihat.npy
python create_nmf_drum_templates_from_sample_library.py --samples-list-file resources/tracklist/templates/samples_crash.csv --output-file resources/templates/crash.npy
```

## Copyright information

Copyright 2020 Len Vande Veire.  

This code within this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.