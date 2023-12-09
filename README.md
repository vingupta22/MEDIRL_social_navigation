# MEDIRL_social_navigation
2nd Place Winner for ML@Purdue Advanced Track Machine Learning Hackathon.
All code and experiments available via my DagsHub Repo here: https://dagshub.com/ML-Purdue/hackathonf23-Stacks

 ML-Purdue / hackathonf23-Stacks
 
 
...
README.md
11 KB
hackathonf23-Stacks
Runnable code for the project exists in the following project: https://dagshub.com/ML-Purdue/hackathonf23-Stacks The example input for our repo is located in this link: https://dagshub.com/ML-Purdue/hackathonf23-Stacks/src/main/data under train.csv.
The following files include all relevant configuration parameters for the project: requirements.txt & config.yaml
Data (+ Artifacts) All project data and artifacts can be found here: https://dagshub.com/ML-Purdue/hackathonf23-Stacks/src/main/data. The structure/schema of input data is the following: time [ms],person id,position x [m],position y [m],position z (height) [m],velocity [m/s],angle of motion [rad],facing angle [rad]
Data preprocessing:

data = pd.read_csv(file_path)  # Load CSV data
        scaler = MinMaxScaler()
        columns_to_normalize = ['position x [m]', 'position y [m]', 'position z (height) [m]', 'velocity [m/s]']
        data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
        positions = data[['position x [m]', 'position y [m]', 'position z (height) [m]']].values
        velocities = data['velocity [m/s]'].values
Model training:

def train_irl_with_dataset(self, data, lr=0.001, epochs=3):
        state_dim = self.state_dim
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        positions = data[['position x [m]', 'position y [m]', 'position z (height) [m]']].values
        velocities = data['velocity [m/s]'].values
        mlflow.tensorflow.autolog()
        with mlflow.start_run(experiment_id=get_or_create_experiment_id("Base Reproduction")):
            
            for epoch in range(epochs):
                total_loss = 0
                state_frequencies = self._calculate_state_frequencies(positions)

                for idx in range(len(positions)):
                    state = positions[idx]
                    velocity = velocities[idx]

                    with tf.GradientTape() as tape:
                        preferences = self.model(state[np.newaxis, :])
                        prob_human = tf.nn.softmax(preferences)

                        # Define losses
                        max_entropy_loss = -tf.reduce_sum(prob_human * tf.math.log(prob_human + 1e-8), axis=1)
                        alignment_loss = -tf.reduce_sum(state_frequencies * tf.math.log(prob_human + 1e-8), axis=1)
                        maxent_irl_objective = max_entropy_loss + alignment_loss

                        # Compute the gradients
                        grads = tape.gradient(maxent_irl_objective, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                        total_loss += tf.reduce_sum(maxent_irl_objective)  # Accumulate the total loss

                avg_loss = total_loss / len(positions)
                mlflow.log_metric(f"loss", avg_loss, step=epoch)
                print(f"Epoch {epoch + 1}/{epochs}, MaxEnt IRL Loss: {avg_loss}")
Outputs: https://dagshub.com/ML-Purdue/hackathonf23-Stacks/experiments/#/

Environment Software package documentation located in: requirements.txt The model was trained and tested on a 2018 MacBook Pro, with an Intel i7 chip, an Intel UHD Graphics 630 GPU, along with a Radeon Pro 560X discrete GPU from AMD with 4GB of GDDR5 memory

Evaluation

Instructions to Run Model Tests:
Instructions to test Base Reproduction
To evaluate the BASE reproduction:

open 'reproduced_model_runner.py' and change the following directories:
"test_data_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl'"
to:
"test_data_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/models/reproduced_model.pkl'" respectively.

then, run 'python3 reproduced_model_runner'
Instructions to test Removed Hidden Layer Ablation
To evaluate the Removed Hidden Layer reproduction:

open 'removed_hidden_layer_runner.py' and change the following directories:
"test_data_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl'"
to:
"test_data_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/models/removed_hidden_layer.pkl'" respectively.

then, run 'python3 removed_hidden_layer_runner.py'
Instructions to test Removed Maximum Entropy Ablation
To evaluate the Removed Maximum Entropy reproduction:

open 'rmv_max_entropy_runner.py' and change the following directories:
"test_data_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl'"
to:
"test_data_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/models/rmv_max_entropy_model.pkl'" respectively.

then, run 'python3 rmv_max_entropy_runner.py'
Instructions to test Removed Discount Factor Ablation
To evaluate the Removed Discount Factor reproduction:

open 'rmv_discount_runner.py' and change the following directories:
"test_data_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl'"
to:
"test_data_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/models/rmv_discount_model.pkl'" respectively.

then, run 'python3 rmv_discount_runner.py'
Instructions to test Removed State Dimension Ablation
To evaluate the Removed State Dimension reproduction:

open 'rmv_dimension_runner.py' and change the following directories:
"test_data_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl'"
to:
"test_data_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/models/rmv_dim_model.pkl'" respectively.

then, run 'python3 rmv_dimension_runner.py'
Instructions to test Leaky ReLU Ablation
To evaluate the Removed Leaky ReLU reproduction:

open 'leaky_relu_runner.py' and change the following directories:
"test_data_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl'"
to:
"test_data_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/test.csv'" and 
"model_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/models/leaky_relu_model.pkl'" respectively.

then, run 'python3 leaky_relu_runner.py'
Instructions to Retrain Models (not necessary to test, but if you chose to do so):
Instructions to re-train Base Reproduction:
To re-train the BASE reproduction:

open 'reproduced_model.py' and change the following directories:
"file_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl')"

to:
"file_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/your/path/to/our/local/repository/hackathonf23-Stacks/models/reproduced_model.pkl')'" respectively.

then, run 'python3 reproduced_model.py'
Instructions to re-train Removed Hidden Layer Ablation:
To re-train the Removed Hidden Layer Ablation:

open 'removed_hidden_layer.py' and change the following directories:
"file_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl')"

to:
"file_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/your/path/to/our/local/repository/hackathonf23-Stacks/models/removed_hidden_layer.pkl')'" respectively.

then, run 'python3 removed_hidden_layer.py'
Instructions to re-train Removed Max Entropy Ablation:
To re-train the Removed Max Entropy Ablation:

open 'rmv_max_entropy_local.py' and change the following directories:
"file_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl')"

to:
"file_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/your/path/to/our/local/repository/hackathonf23-Stacks/models/rmv_max_entropy_model.pkl')'" respectively.

then, run 'python3 rmv_max_entropy_local.py'
Instructions to re-train Removed Discount Factor Ablation:
To re-train the Removed Discount Factor Ablation:

open 'rmv_discount_local.py' and change the following directories:
"file_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl')"

to:
"file_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/your/path/to/our/local/repository/hackathonf23-Stacks/models/rmv_discount_model.pkl')'" respectively.

then, run 'python3 rmv_discount_local.py'
Instructions to re-train Removed State Dimension Ablation:
To re-train the Removed State Dimension Ablation:

open 'rmv_dim.py' and change the following directories:
"file_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/reproduced_model.pkl')"

to:
"file_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/your/path/to/our/local/repository/hackathonf23-Stacks/models/rmv_dim_model.pkl')'" respectively.

then, run 'python3 rmv_dim.py'
Instructions to re-train Leaky ReLU Ablation:
To re-train the Leaky ReLU Ablation:

open 'leaky_relu_local.py' and change the following directories:
"file_path = '/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/Users/vinay/Desktop/Computer_Science_Projects/ReScience/hackathonf23-Stacks/models/leaky_relu_model.pkl')"

to:
"file_path = '/your/path/to/our/local/repository/hackathonf23-Stacks/data/train.csv'" and 
"irl.save_model('/your/path/to/our/local/repository/hackathonf23-Stacks/models/reproduced_model.pkl')'" respectively.

then, run 'python3 leaky_relu_local.py'
 
