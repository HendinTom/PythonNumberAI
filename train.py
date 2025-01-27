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
            d_avg_cost_to_bias1 = np.zeros_like(model.biases1)

            d_avg_cost_to_weights2 = np.zeros_like(model.weights2)
            d_avg_cost_to_bias2 = np.zeros_like(model.biases2)

            d_avg_cost_to_weights3 = np.zeros_like(model.weights3)
            d_avg_cost_to_bias3 = np.zeros_like(model.biases3)

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
                y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                y[label] = 1
                
                # Compute loss
                cost = np.mean((output - y) ** 2)
                total_cost += cost

                #Get which ones it got right
                if prediction == label:
                    correct += 1
                
                # Backpropagation `d` - stands for derivative
                d_cost_to_a3 = 2*(output - y) #What 2(a3 -y) looks like in code

                d_z3_to_a3 = sig_derivative(model.z3)
                d_weights3_to_z3 = model.a2
                d_cost_to_weights3 = (d_weights3_to_z3[:, np.newaxis] * (d_z3_to_a3 * d_cost_to_a3))
                d_avg_cost_to_weights3 += d_cost_to_weights3
                # print("Cost to weights3:", d_cost_to_weights3.shape)
                d_cost_to_bias3 = (d_z3_to_a3 * d_cost_to_a3).reshape(1, -1)
                d_avg_cost_to_bias3 += d_cost_to_bias3
                # print("Cost to bias3:", d_cost_to_bias3.shape)

                d_cost_to_a2 = np.sum((model.weights3.T * (d_z3_to_a3 * d_cost_to_a3).reshape(-1, 1)).T, axis=1).reshape(1, -1)
                # print(d_cost_to_a2.shape)

                d_z2_to_a2 = sig_derivative(model.z2)
                d_weights2_to_z2 = model.a1
                d_cost_to_weights2 = (d_weights2_to_z2[:, np.newaxis] * (d_z2_to_a2 * d_cost_to_a2))
                d_avg_cost_to_weights2 += d_cost_to_weights2
                # print(d_cost_to_weights2.shape)
                d_cost_to_bias2 = (d_z2_to_a2 * d_cost_to_a2).reshape(1, -1)
                d_avg_cost_to_bias2 += d_cost_to_bias2
                # print(d_cost_to_bias2.shape)

                d_cost_to_a1 = np.sum((model.weights2.T * (d_z2_to_a2 * d_cost_to_a2).reshape(-1, 1)).T, axis=1).reshape(1, -1)
                # print(d_cost_to_a1.shape)
                
                d_z1_to_a1 = sig_derivative(model.z1)
                d_weights1_to_z1 = input_data
                d_cost_to_weights1 = (d_weights1_to_z1[:, np.newaxis] * (d_z1_to_a1 * d_cost_to_a1))
                d_avg_cost_to_weights1 += d_cost_to_weights1
                # print(d_cost_to_weights1.shape)
                d_cost_to_bias1 = (d_z1_to_a1 * d_cost_to_a1).reshape(1, -1)
                d_avg_cost_to_bias1 += d_cost_to_bias1
                # print(d_cost_to_bias1.shape)

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
    save_model(model)