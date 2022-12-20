# Movie Recommendation system application using Taipy
- In the application, the user's information is secured as `user_id`.
- The app recommends **top_k** movies for a user via user_id, also give the **top_k actual** movies that the user is interested in.
- There are 2 models used to make recommendations: *kNN* and *MF*
- There are two metrics to evaluate the performance of the model: *recall* and *precision*. These are calculated automatically after making recommendations

## Run the application on a local machine
- Step 1: Clone the source code for Taipy application.
On your command prompt, run <br/>
<code>git clone https://github.com/caotuannghia20/recsys-taipy-app.git </code>
 - Step 2: Download the MovieLens 100K dataset [here](https://grouplens.org/datasets/movielens/100k/).
 - Step 3: Split file `u.data` into training (**rating_train.csv**) and testing (**rating_test.csv**) data.
First, edit the `TEST_RATIO` and `MOVIELENS_DATA_PATH` on `ultis/splitdata.py` then run:
```
python utils/splitdata.py
```
 - Step 4: Edit the dataset path in `config/config.py` line 22, 23, 24
 - Step 5: Install the requirements enviroment, type : <br/>
```
pip install -r requirements.txt
```
 - Step 6: On your command prompt, run <br/>
```
python main.py
 ``` 
<br/> to run the application on a local machine
## Use the application
After run the application:

 We choose the model to make recommendations
![Choose the model to make recommendations](image/demo1.png )

Suppose to choose the model kNN, the display shows the parameters of the kNN model, we adjust or keep the default value. And then click **CREATE_SCENARIO** to sumbit the parameters and create model, 
similar to the MF model.

![Adjust the parameters](image/demo2.png )
After create model, the prediction section will appear.
![The prediction section](image/demo3.png )
Adjust the **top_k** movies want to recommend and click **RECOMMEND** to show the results. The results include **top_k** movies recommendations for a user via user_id, the **top_k actual** movies that the user is interested in, the metrics ranking *recall* and *precision*.
![Results](image/demo4.png )

# Notice:
If you want to change the model or model's parameters, you adjuts and click the **SAVE CHANGES** button.