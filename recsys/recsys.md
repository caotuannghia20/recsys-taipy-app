# MOVIE REOMMENDATION SYSTEM
#Choose the model
<|{selected_model}|selector|lov={model_selector}|width = 100|dropdown|>
<|part|render={selected_model == "MF"}|
# Creat your scenario and change your hyper parameters
<|layout|columns=1 1 1 1 1|
<|
**n_epochs** <br/> <|{n_epochs}|number|>
|>

<|
**n_factors** <br/> <|{n_factors}|number|>
|>

<|
**learning_rate** <br/> <|{learning_rate}|number|>
|>

<|
**x_id** <br/> <|{x_id}|number|>
|>

<|
<br/> <|create_scenario|button|on_action={create_scenario}|>
|>
|>
|>
<|part|render={len(scenario_selector) > 0 and selected_model == "MF" }|
<|layout|columns=1|
<|
##  <|Save changes|button|on_action={submit_scenario}|>
|>

|>

Press <|recommend|button|on_action={predicts}|>
<|layout|columns=1 1|
<|
To recommend top <|{top_k}|> movies to user <|{x_id+1}|>: <br/> <|{y_id}|tree|lov={results}|width = "400px"|> 
|>

<|
And these are the real movies the user <|{x_id+1}|> want to watch: <br/>
<|{y_id_real}|tree|lov={results_real}|width = "400px"|> 
|>

<|
**top_k** <br/> <|{top_k}|number|>
|>
|>

<|layout|columns=1 1|
<|
**Precision** = <|{precision}|> 
|>

<|
**Recall** = <|{recall}|> 
|>
|>
|>
<|part|render={selected_model == "kNN"}|
# Creat your scenario and change your hyper parameters
<|layout|columns=1 1 1 1|
<|
**k_neighboor** <br/> <|{n_k_neighboor}|number|>
|>

<|
**k_min** <br/> <|{n_min_k}|number|>
|>

<|
**x_id** <br/> <|{x_id}|number|>
|>

<|
<br/> <|create_scenario|button|on_action={create_scenario}|>
|>
|>

#Select the similarity measure
<|{selected_sim_measure}|selector|lov={sim_measure_selector}|width = 100|>
|>

<|part|render={len(scenario_selector) > 0 and selected_model == "kNN" }|
<|layout|columns=1|
<|
##  <|Save changes|button|on_action={submit_scenario}|>
|>

|>

Press <|recommend|button|on_action={predicts}|>
<|layout|columns=1 1|
<|
To recommend top <|{top_k}|> movies to user <|{x_id+1}|>: <br/> <|{y_id}|tree|lov={results}|width = "400px"|> 
|>

<|
And these are the real movies the user <|{x_id+1}|> want to watch: <br/>
<|{y_id_real}|tree|lov={results_real}|width = "400px"|> 
|>

<|
**top_k** <br/> <|{top_k}|number|>
|>
|>

<|layout|columns=1 1|
<|
**Precision** = <|{precision}|>
|>

<|
**Recall** = <|{recall}|> 
|>
|>
|>