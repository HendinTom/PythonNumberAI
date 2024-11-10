import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from simple_nn import SimpleNN

# np.set_printoptions(threshold=np.inf)

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_derivative(x):
    return sig(x) * (1 - sig(x))

# Load MNIST data
def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return train_data

def train(model, train_data, epochs=5, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for i, (image, label) in enumerate(train_data):
            # Flatten image to 784x1 vector and normalize pixel values
            input_data = image.view(784).numpy()
            input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
            
            # Forward pass
            output, _ = model.forward(input_data)
            
            # One-hot encode the label
            target = np.zeros(10)
            target[label] = 1
            
            # Compute loss (Mean Squared Error)
            loss = np.mean((output.flatten() - target) ** 2)
            total_loss += loss
            
            # Backpropagation
            output_error = output.flatten() - target
            
            # Gradients for weights and biases of each layer
            d_weights4 = np.dot(output_error.reshape(-1, 1), model.a3.T.reshape(1, -1))  # Shape (10, 16)
            d_biases4 = output_error
            
            hidden_error3 = (np.dot(model.weights4.T, output_error) * model.a3 * (1 - model.a3)).flatten()  # Flatten to shape (16,)
            d_weights3 = np.dot(hidden_error3.reshape(-1, 1), model.a2.T.reshape(1, -1))  # Shape (16, 16)
            d_biases3 = hidden_error3

            hidden_error2 = (np.dot(model.weights3.T, output_error) * model.a2 * (1 - model.a2)).flatten()  # Flatten to shape (16,)
            d_weights2 = np.dot(hidden_error2.reshape(-1, 1), model.a1.T.reshape(1, -1))  # Shape (16, 16)
            d_biases2 = hidden_error2
            
            hidden_error1 = (np.dot(model.weights2.T, hidden_error2) * model.a1 * (1 - model.a1)).flatten()  # Flatten to shape (16,)
            d_weights1 = np.dot(hidden_error1.reshape(-1, 1), input_data.reshape(1, -1))  # Shape (16, 784)
            d_biases1 = hidden_error1

            # Gradient descent update
            model.weights4 -= learning_rate * d_weights4
            model.biases4 -= learning_rate * d_biases4[:, None]
            model.weights3 -= learning_rate * d_weights3
            model.biases3 -= learning_rate * d_biases3[:, None]
            model.weights2 -= learning_rate * d_weights2
            model.biases2 -= learning_rate * d_biases2[:, None]
            model.weights1 -= learning_rate * d_weights1
            model.biases1 -= learning_rate * d_biases1[:, None]
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}")


# Save the model parameters
def save_model(model, filename="trained_model.pkl"):
    model_params = {
        "weights1": model.weights1,
        "biases1": model.biases1,
        "weights2": model.weights2,
        "biases2": model.biases2,
        "weights3": model.weights3,
        "biases3": model.biases3,
        "weights3": model.weights4,
        "biases3": model.biases4,
    }
    with open(filename, "wb") as f:
        pickle.dump(model_params, f)
    print(f"Model saved to {filename}")

# Main training procedure
if __name__ == "__main__":
    train_data = load_mnist_data()
    model = SimpleNN()
    train(model, train_data, epochs=5, learning_rate=0.01)
    save_model(model)

    # train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # #One loop of this is one batch of training data
    # for images, labels in train_loader:
    #     correct_guess = 0
    #     batch_cost_layer_4 = 0 # !Might remove because it is unessary
    #     # dir_batch_cost_layer_4 = 0

    #     # dir_batch_z3 = 0

    #     # dir_batch_weights3 = 0

    #     dir_batch_weights1_result = np.array([[0] * 784] * 16)
    #     dir_batch_bias1_result = np.array([[0]] * 16)

    #     dir_batch_weights2_result = np.array([[0] * 16] * 16)
    #     dir_batch_bias2_result = np.array([[0]] * 16)

    #     dir_batch_weights3_result = np.array([ [0] * 16 ] * 10) # Creating an array that has 16 lists and each list has 10 elements to match the shape of the last layer
    #     dir_batch_bias3_result = np.array([[0]] * 10) # Creating an array that has 10 items to match the bias in the last layer
        
    #     # Flatten each 28x28 image to a 784-dimensional vector
    #     images = images.view(images.size(0), -1)  # Reshape to [batch_size, 784]

    #     # Convert images from [-1, 1] range to [0, 1] range
    #     images = (images + 1) / 2

    #     # Process each `images` batch with your model
    #     outputs = model.forward(images)  # Forward pass through the model

    #     batch = outputs[0] #This contains the last layer for one batch in the model
    #     batch_predictions = outputs[1] #This is the prediction the model makes for one batch

    #     batch_a1 = model.a1
    #     batch_z1 = model.z1
    #     batch_a2 = model.a2
    #     batch_z2 = model.z2
    #     batch_z3 = model.z3
        
    #     for x in range(len(batch)):
    #         last_layer = batch[x] #? IS EQUIVALENT TO a3 SO I WILL BE USING THAT AS A REPLACEMENT FOR IT
    #         prediction = batch_predictions[x]
    #         label = labels[x]
    #         if(prediction == label):
    #             correct_guess += 1
    #         print("")
    #         # print("Last Layer Activations:", last_layer)
    #         print("Prediction of Model:", prediction)
    #         print("Actual Answer:", label)

    #         #Getting the cost of the first layer (adding up the squares of the differences in the last layer to the label)
    #         y = np.array([0] * 10) #Making an array of 10 zeros (using numpy because it is fast with the tradoff that I cannot change the array size)
    #         y[label] = 1 #Making the label which should be the index of the highest activation to 1 in the array

    #         #!FIX THE SHAPES OF THE FIRST LAYER WHEN TRYING TO TRAIN
    #         """
    #         ALL THINGS TO DO WITH THE COST
    #         """
    #         cost_arr = np.subtract(last_layer, y) # Getting the cost array (the squares of the differences in the 10 activations)
    #         cost = np.sum(np.square(cost_arr)) #The actual cost of the prediction is the sum of all those numbers in the cost array MIGHT REMOVE BECUASE UNESSARY
    #         batch_cost_layer_4 += cost #Adding the cost to the total cost of the batch                                              MIGHT REMOVE BECAUSE UNESSARY

    #         dir_cost = np.multiply(2, cost_arr) # The dirivative of the cost being 2(activation - answer_activation) and the sum of that over the 10 activations in the last layer
    #         # dir_batch_cost_layer_4 += dir_cost # Adding the dirivative of the cost to the dirivative cost of the batch

    #         # print("array of cost:", cost_arr)
    #         # print("dirivative of cost:", dir_cost)
    #         # print("Shape of derivative of cost:", dir_cost.shape)
    #         # print("cost:", cost)

    #         """
    #         ALL THINGS TO DO WITH THE SIGMOID FOR LAYER 3
    #         """
    #         z3 = batch_z3[x] #z3 is a list of the last layer activations before the sigmoid function was applied
    #         dir_z3 = sig_derivative(z3) # Getting the derivative of z3 through the sigmoid
    #         # dir_batch_z3 += dir_z3 # Adding it to the batch of derivative of z3

    #         # print("z3", z3)
    #         # print("dirivative of z3", dir_z3)
    #         # print("Shape of derivative of z3:", dir_z3.shape)

    #         """
    #         ALL THINGS TO DO WITH THE WEIGHTS FOR LAYER 3
    #         """
    #         a2 = batch_a2[x] # When you work out the derivative of the weights, it comes out to the activation in the previous layer
    #         a2_reshaped = np.tile(a2.reshape(1, -1), (10, 1))  # Shape becomes (10, 16)
    #         dir_z3_reshaped = dir_z3.reshape(10, 1)    # Shape becomes (1, 10)
    #         dir_cost_reshaped = dir_cost.reshape(10, 1)  # Shape becomes (1, 10)
    #         weights3_result = a2_reshaped * dir_z3_reshaped * dir_cost_reshaped  # Final shape is (10, 16)

    #         dir_batch_weights3_result = np.add(dir_batch_weights3_result, weights3_result)

    #         # print("Shape of a2:", a2.shape)
    #         # print("Shape of a2_reshaped:", a2_reshaped)
    #         # print("Shape of dir_z3_reshaped:", dir_z3_reshaped.shape)

    #         # testing = np.array([list(range(0,16))] * 10)
    #         # testing2 = np.array([10] * 9 + [12]).reshape(10, 1)
    #         # print(testing * testing2)
    #         """
    #         ALL THINGS TO DO WITH THE BIAS FOR LAYER 3
    #         """
    #         bias3_result = (dir_z3_reshaped * dir_cost_reshaped).reshape(10, 1)
    #         dir_batch_bias3_result = np.add(dir_batch_bias3_result, bias3_result)

    #         # print("Bias3 result shape:", bias3_result.shape)
    #         """
    #         ALL THINGS TO DO WITH THE PREVIOUS LAYER (Layer 2 of 3)
    #         """
    #         weights3 = model.weights3.T
    #         dir_a2 = (weights3.sum(axis=1) * dir_z3_reshaped * dir_cost_reshaped).sum(axis=0)
            
    #         # print("dir of a2:", dir_a2)
    #         # print("shape of dir of a2:", dir_a2.shape)

    #         """
    #         ALL THINGS TO DO WITH THE SIGMOID OF LAYER 2
    #         """
    #         z2 = batch_z2[x]
    #         dir_z2 = sig_derivative(z2)

    #         # print("dir shape of z2:", dir_z2.shape)
    #         """
    #         ALL THINGS TO DO WITH THE WEIGHTS OF LAYER 2
    #         """
    #         a1 = batch_a1[x]
    #         a1_reshaped = np.tile(a1.reshape(1, -1), (16, 1)) #Shape becomes (16, 16)
    #         dir_z2_reshaped = dir_z2.reshape(16, 1) #Shape becomes (1, 16)
    #         dir_a2_reshaped = dir_a2.reshape(16, 1) #Shape becomes (16, 1)
    #         weights2_result = a1_reshaped * dir_z2_reshaped * dir_a2_reshaped

    #         dir_batch_weights2_result = np.add(dir_batch_weights2_result, weights2_result)

    #         # print("dir_z2_reshaped:", dir_z2_reshaped.shape)
    #         # print("Weights2 result:", weights2_result)
    #         # print("Weights2 shape:", weights2_result.shape)
    #         """
    #         ALL THINGS TO DO WITH THE BIASES OF LAYER 2
    #         """
    #         bias2_result = (dir_z2_reshaped * dir_a2_reshaped).reshape(16, 1)
    #         dir_batch_bias2_result = np.add(dir_batch_bias2_result, bias2_result)

    #         # print("Bias2 result:", bias2_result)
    #         # print("Bias2 result shape:", bias2_result.shape)

    #         """
    #         ALL THINGS TO DO WITH THE PREVIOUS PREVIOUS LAYER (LAYER 1 OF 3)
    #         """
    #         weights2_16x16 = model.weights2.T
    #         dir_a1 = (weights2_16x16.sum(axis=1) * dir_z2_reshaped * dir_a2.reshape(16, 1)).sum(axis=0)

    #         # print("dir of a1:", dir_a1)
    #         # print("shape of dir of a1:", dir_a1.shape)

    #         """
    #         ALL THINGS TO DO WTIH THE SIGMOID OF LAYER 1
    #         """
    #         z1 = batch_z1[x]
    #         dir_z1 = sig_derivative(z1)

    #         # print("dir shape of z1:", dir_z1.shape)

    #         """
    #         ALL THINGS TO DO WITH THE WEIGHTS OF LAYER 1
    #         """
    #         image_input = images[x]
    #         image_input_reshaped = np.tile(image_input.reshape(1, -1), (16, 1)) #Shape is (16, 784)
    #         dir_z1_reshaped = dir_z1.reshape(16, 1) #!Make sure to change the others to be like: Shape becomes (16, 1)
    #         dir_a1_reshaped = dir_a1.reshape(16, 1) #Shape is (16, 1)
    #         weights1_result = image_input_reshaped * dir_z1_reshaped * dir_a1_reshaped

    #         dir_batch_weights1_result = np.add(dir_batch_weights1_result, weights1_result)

    #         # print("Shape of image_input_reshaped:", image_input_reshaped.shape)
    #         # print("Shape of dir_z1_reshaped:", dir_z1_reshaped.shape)
    #         # print("Shape of weights1:", weights1_result.shape)

    #         """
    #         ALL THINGS TO DO WITH THE BIAS OF LAYER 1
    #         """
    #         bias1_result = (dir_z1_reshaped * dir_a1_reshaped).reshape(16, 1)
    #         dir_batch_bias1_result = np.add(dir_batch_bias1_result, bias1_result)

    #         # print("Shape of bias1:", bias1_result.shape)

    #     print("Accuracy:", correct_guess/len(batch)*100, "%")

    #     dir_batch_weights1_result = dir_batch_weights1_result/len(batch)
    #     dir_batch_bias1_result = dir_batch_bias1_result/len(batch)

    #     dir_batch_weights2_result = dir_batch_weights2_result/len(batch)
    #     dir_batch_bias2_result = dir_batch_bias2_result/len(batch)
        
    #     dir_batch_weights3_result = dir_batch_weights3_result/len(batch)
    #     dir_batch_bias3_result = dir_batch_bias3_result/len(batch)

    #     # print("dir Weight3 results:", dir_batch_weights3_result)
    #     # print("dir bias3 results:", dir_batch_bias3_result)
        
    #     # dir_batch_weights3 = dir_batch_weights3/len(batch) #Taking the average across the training examples for the derivative of weights3

    #     # dir_batch_z3 = dir_batch_z3/len(batch) # Taking the average across the training examples for the the dirivative of z3
        
    #     # dir_batch_cost_layer_4 = dir_batch_cost_layer_4/len(batch) # Taking the average across the training examples for the dirivative of the cost
    #     batch_cost_layer_4 = batch_cost_layer_4/len(batch) # Taking the average across the training examples for the cost

    #     # print("Dirivative Batch Cost:", dir_batch_cost_layer_4)
    #     print("Batch Cost:", batch_cost_layer_4)
    #     # `labels` contains the ground truth digit (0-9) for each image
    #     print("Batch labels:", labels)  # Labels for the batch (tensor of integers)
    #     # print("\n")

    #     # print(model.weights1.shape)
    #     # print((learning_rate*dir_batch_weights1_result).shape)
    #     # print("weights1 shape:", dir_batch_weights1_result.shape)
    #     # print("bias1 shape:", dir_batch_bias1_result.shape)

    #     # print("weights2 shape:", dir_batch_weights2_result.shape)
    #     # print("bias2 shape:", dir_batch_bias2_result.shape)

    #     # print("weights3 shape:", dir_batch_weights3_result.shape)
    #     # print("bias3 shape:", dir_batch_bias3_result.shape)
    #     # print("\n")

    #     # print("weights1 shape:", model.weights1.shape)
    #     # print("bias1 shape:", model.biases1.shape)

    #     # print("weights2 shape:", model.weights2.shape)
    #     # print("bias2 shape:", model.biases2.shape)

    #     # print("weights3 shape:", model.weights3.shape)
    #     # print("bias3 shape:", model.biases3.shape)
    #     # print("\n")

    #     model.biases1 = np.subtract(model.biases1, learning_rate * dir_batch_bias1_result)
    #     model.weights1 = np.subtract(model.weights1, learning_rate * dir_batch_weights1_result)

    #     model.biases2 = np.subtract(model.biases2, learning_rate * dir_batch_bias2_result)
    #     model.weights2 = np.subtract(model.weights2, learning_rate * dir_batch_weights2_result)
        
    #     model.biases3 = np.subtract(model.biases3, learning_rate * dir_batch_bias3_result)
    #     model.weights3 = np.subtract(model.weights3, learning_rate * dir_batch_weights3_result)

    #     # print("weights1 shape:", model.weights1.shape)
    #     # print("bias1 shape:", model.biases1.shape)

    #     # print("weights2 shape:", model.weights2.shape)
    #     # print("bias2 shape:", model.biases2.shape)

    #     # print("weights3 shape:", model.weights3.shape)
    #     # print("bias3 shape:", model.biases3.shape)

    #     # model.biases1 = model.biases1.flatten()
    #     # model.biases2 = model.biases2.flatten()
    #     # model.biases3 = model.biases3.flatten()
    #     # break