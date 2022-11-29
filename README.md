# NollaFraud

### System tested
Ubuntu 20.04.5 LTS

### Configurations
<div>Download the repository</div> 
<code>git clone "https://github.com/C1rF/NollaFraud/edit/master/README.md"</code>
<br/>

config the environment: <br/>
install the following packages: <br/>
<code>
python==3.9.0 
</code>
<br/>
<code>
numpy==1.21.0
</code>
<br/>
<code>
scikit-learn==0.23.2
</code>
<br/>
<code>
tensorflow==2.10.0
</code>
<br/>
<code>
keras-tuner==1.1.3
</code>

To config through pip from <code>requirements.txt</code><br/>
To config through conda from <code>environment.yml</code>

### Run in Graph Mode
<b>Code in 'master' branch</b>: train NollaFraud model on Amazon dataset using graph mode execution

<b>run</b> <code>python train.py</code>

### Run in Eager Execution Mode
<b>Code in 'eager-exec' branch</b>: train NollaFraud model on Amazon dataset using eager mode execution

<b>run</b> <code>python train_then_eval.py</code>

### <b>hyperparameter tuning</b>:
run <code>python tuner.py</code>
results will save in <code>./customTunerResult/hypertuning/</code>

### File Description
<code>train.py</code>: The training entrance of graph mode under master branch <br/>
<code>train_then_eval.py</code>: The training entrance of eager mode under eager-exec branch <br/>
<code>layers.py</code>: The definition of the NollaFraud custom model, and each custom layers used in the custom model. <br/>
<code>best_hps.txt</code>: Tuned hyperparameters. <br/>
<code>customTunerResult/</code>: Results of 15 hyperparameter tuning trials.
