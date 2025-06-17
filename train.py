import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from simple_nn import SimpleNN

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_derivative(x):
    return sig(x) * (1 - sig(x))

# Load MNIST data
def load_mnist_data(batch_size=100):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # Making the data into batches of random numbers from the training set
    train_data = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    return train_data

def train(model, train_data, epochs=5, learning_rate=0.01):
    for epoch in range(epochs):
        total_cost = 0
        correct = 0
        #Going through each batch
        for batch_idx, (data, targets) in enumerate(train_data):
            # Initialize gradients to zero for each batch
            d_avg_cost_to_weights1 = np.zeros_like(model.weights1)
            d_avg_cost_to_weights1Mw = np.zeros_like(model.weights1Mw)
            d_avg_cost_to_weights1Mr = np.zeros_like(model.weights1Mr)
            d_avg_cost_to_bias1 = np.zeros_like(model.biases1)
            d_avg_cost_to_bias1Mw = np.zeros_like(model.biases1Mw)
            d_avg_cost_to_bias1Mr = np.zeros_like(model.biases1Mr)

            d_avg_cost_to_weights2 = np.zeros_like(model.weights2)
            d_avg_cost_to_weights2Mw = np.zeros_like(model.weights2Mw)
            d_avg_cost_to_weights2Mr = np.zeros_like(model.weights2Mr)
            d_avg_cost_to_bias2 = np.zeros_like(model.biases2)
            d_avg_cost_to_bias2Mw = np.zeros_like(model.biases2Mw)
            d_avg_cost_to_bias2Mr = np.zeros_like(model.biases2Mr)

            d_avg_cost_to_weights3 = np.zeros_like(model.weights3)
            d_avg_cost_to_weights3Mw = np.zeros_like(model.weights3Mw)
            d_avg_cost_to_weights3Mr = np.zeros_like(model.weights3Mr)
            d_avg_cost_to_bias3 = np.zeros_like(model.biases3)
            d_avg_cost_to_bias3Mw = np.zeros_like(model.biases3Mw)
            d_avg_cost_to_bias3Mr = np.zeros_like(model.biases3Mr)

            #Going through each image
            for i in range(len(data)):
                image = data[i]
                label = targets[i]

                # Flatten image and normalize the image to (784,)
                input_data = image.view(784).numpy()
                input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

                #*This section here is just if you want to conduct some tests on how arrays get multiplied together and how to shape them to do what you want them to do
                # test = np.array([[1, 2], [3, 4], [5, 6]]) #Flipping it so that it can be multiplied with the correct numbers mulitplying
                # test1 = np.array([7, 8, 9])
                # print(test.shape)
                # test2 = np.sum((test.T * test1), axis=0)
                # print(test2) #Flipping it back so that it returns to the correct shape

                # Forward pass
                output, prediction = model.forward(input_data)

                # Making an array for the y value and setting the correct index to 1
                y = np.zeros(10)
                y[label] = 1

                # Compute loss
                cost = np.mean((output - y) ** 2)
                total_cost += cost

                # Get which ones it got right
                if prediction == label:
                    correct += 1

                # Backpropagation
                #!Finished calculations
                #*Third Layer
                d_cost_to_a3 = 2 * (output - y)  # shape: (1, 10)
                d_za3_to_a3 = sig_derivative(model.za3) #za3A because that is when the bias is added

                #Reading Memory Parameters: Layer 3
                d_cost_to_weights3Mr = model.memory * d_za3_to_a3.T * d_cost_to_a3.T #*Make sure this multiplies out correctly
                d_cost_to_bias3Mr = 1 * np.sum(d_za3_to_a3 * d_cost_to_a3.T, axis=1)

                d_avg_cost_to_weights3Mr += d_cost_to_weights3Mr
                d_avg_cost_to_bias3Mr += d_cost_to_bias3Mr

                #Writing Memory Parameters: Layer 3
                d_memory_to_za3 = model.weights3Mr
                # Rechecking if this works because it doesn't in layer 2
                print("model.z3.T shape:", model.z3.shape)
                print("d_memory_to_za3 shape:", d_memory_to_za3.shape)
                print("d_za3_to_a3.T shape:", d_za3_to_a3.shape)
                print("d_cost_to_a3.T shape:", d_cost_to_a3.shape)

                d_cost_to_weights3Mw = model.z3.T * d_memory_to_za3 * d_za3_to_a3.T * d_cost_to_a3.T
                d_cost_to_bias3Mw = 1 * d_memory_to_za3.T * d_za3_to_a3 * d_cost_to_a3

                d_avg_cost_to_weights3Mw += d_cost_to_weights3Mw
                d_avg_cost_to_bias3Mw += np.sum(d_cost_to_bias3Mw, axis=1)

                #Last Layer Parameters
                d_z3_to_memory = model.weights3Mw

                # print("der of z3 to memory:", d_z3_to_memory.shape)
                # print("der of za3 to a3:", d_za3_to_a3.shape)
                # print(d_z3_to_memory.shape)
                # print(d_cost_to_a3.shape)
                d_weights3_to_z3 = np.tile(model.a2.flatten(), (10, 1)).T
                d_cost_to_weights3 = d_weights3_to_z3 * np.sum(d_z3_to_memory, axis=1).T * np.sum(d_memory_to_za3, axis=1).T * d_za3_to_a3 * d_cost_to_a3
                d_cost_to_bias3 = 1 * np.sum(d_z3_to_memory, axis=1).T * np.sum(d_memory_to_za3, axis=1).T * d_za3_to_a3 * d_cost_to_a3

                d_avg_cost_to_weights3 += d_cost_to_weights3
                d_avg_cost_to_bias3 += d_cost_to_bias3

                #*Second Layer
                d_cost_to_a2 = model.weights3 * np.sum(d_z3_to_memory, axis=1).T * np.sum(d_memory_to_za3, axis=1).T * d_za3_to_a3
                d_za2_to_a2 = sig_derivative(model.za2)

                #Reading Memory Parameters: Layer 2
                A = model.memory2 # shape (16, 100)
                B = d_za2_to_a2.T * d_cost_to_a2  # shape (16, 10)
                B_expanded = np.repeat(B, 100 // B.shape[1], axis=1)  # (16, 100)
                d_cost_to_weights2Mr = A * B_expanded  # (16, 100)
                d_cost_to_bias2Mr = 1 * d_za2_to_a2 * np.sum(d_cost_to_a2, axis=1).T # (1, 16)

                d_avg_cost_to_weights2Mr += d_cost_to_weights2Mr
                d_avg_cost_to_bias2Mr += d_cost_to_bias2Mr

                #Writing Memory Parameters: Layer 2
                d_memory_to_za2 = model.weights2Mr
                d_cost_to_weights2Mw = model.z2.T * d_memory_to_za2 * d_za2_to_a2.T * d_cost_to_a2.T
                d_cost_to_bias2Mw = 1 * d_memory_to_za2.T * d_za2_to_a2 * d_cost_to_a2

                d_avg_cost_to_weights2Mw += d_cost_to_weights2Mw
                d_avg_cost_to_bias2Mw += np.sum(d_cost_to_bias2Mw, axis=1).T

                #Second Last Layer Parameters
                d_z2_to_memory = model.weights2Mw
                d_weights2_to_z2 = np.tile(model.a1.flatten(), (16, 1)).T

                d_cost_to_weights2 = d_weights2_to_z2 * np.sum(d_z2_to_memory, axis=1).T * np.sum(d_memory_to_za2, axis=1).T * d_za2_to_a2 * d_cost_to_a2
                d_cost_to_bias2 = 1 * d_z2_to_memory * d_memory_to_za2 * d_za2_to_a2 * d_cost_to_a2

                d_avg_cost_to_weights2 += d_cost_to_weights2
                d_avg_cost_to_bias2 += d_cost_to_bias2

                #*First Layer
                d_cost_to_a1 = model.weights2 * d_z2_to_memory * d_memory_to_za2 * d_za2_to_a2
                d_za1_to_a1 = sig_derivative(model.za1)

                #Reading Memory Parameters: Layer 1
                d_cost_to_weights1Mr = model.memory * d_za1_to_a1 * d_cost_to_a1
                d_cost_to_bias1Mr = 1 * d_za1_to_a1 * d_cost_to_a1

                d_avg_cost_to_weights1Mr += d_cost_to_weights1Mr
                d_avg_cost_to_bias1Mr += d_cost_to_bias1Mr

                #Writing Memory Parameters: Layer 1
                d_memory_to_za1 = model.weights1Mr

                d_cost_to_weights1Mw = model.z1 * d_memory_to_za1 * d_za1_to_a1
                d_cost_to_bias1Mw = 1 * d_memory_to_za1 * d_za1_to_a1

                d_avg_cost_to_weights1Mw += d_cost_to_weights1Mw
                d_avg_cost_to_bias1Mw += d_cost_to_bias1Mw

                #First Layer Parameters
                d_z1_to_memory = model.weights1Mw

                d_cost_to_weights1 = input_data * d_z1_to_memory * d_memory_to_za1 * d_za1_to_a1
                d_cost_to_bias1 = 1 * d_z1_to_memory * d_memory_to_za1 * d_za1_to_a1

                d_avg_cost_to_weights1 += d_cost_to_weights1
                d_avg_cost_to_bias1 += d_cost_to_bias1

            # Gradient descent update for each layer
            model.weights3 -= learning_rate * d_avg_cost_to_weights3
            model.biases3 -= learning_rate * d_avg_cost_to_bias3
            model.weights2 -= learning_rate * d_avg_cost_to_weights2
            model.biases2 -= learning_rate * d_avg_cost_to_bias2
            model.weights1 -= learning_rate * d_avg_cost_to_weights1
            model.biases1 -= learning_rate * d_avg_cost_to_bias1
    
        print(f"Epoch {epoch + 1}, Loss: {total_cost / len(train_data)}, Accruacy: {round((correct / len(train_data)), 2)}%")

# Save the model parameters
def save_model(model, filename="trained_model.pkl"):
    model_params = {
        "weights1": model.weights1,
        "biases1": model.biases1,
        "weights2": model.weights2,
        "biases2": model.biases2,
        "weights3": model.weights3,
        "biases3": model.biases3
    }
    with open(filename, "wb") as f:
        pickle.dump(model_params, f)

# Main training procedure
if __name__ == "__main__":
    train_data = load_mnist_data(batch_size=100)
    model = SimpleNN()
    train(model, train_data, epochs=40, learning_rate=0.01)
    # save_model(model)