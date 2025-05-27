import pygame
import numpy as np
import pickle
from simple_nn import SimpleNN
from scipy.ndimage import gaussian_filter  # Import Gaussian filter

def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))

# Initialize pygame
pygame.init()

# Constants
GRID_SIZE = 28
CELL_SIZE = 20  # Each cell on the canvas will be 20x20 pixels
CANVAS_SIZE = GRID_SIZE * CELL_SIZE
BRUSH_RADIUS = 2  # Smaller brush radius for thinner strokes
brush_intensity = 0.5  # Lower intensity for softer strokes

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Set up the display
screen = pygame.display.set_mode((CANVAS_SIZE + 200, CANVAS_SIZE))
pygame.display.set_caption("28x28 Pixel Drawing")

# Drawing settings
draw_mode = True
canvas = np.zeros((GRID_SIZE, GRID_SIZE))  # Stores grayscale pixel values between 0 and 1

# Function to draw buttons
def draw_buttons():
    pygame.draw.rect(screen, GRAY, (CANVAS_SIZE + 20, 20, 160, 40))
    pygame.draw.rect(screen, GRAY, (CANVAS_SIZE + 20, 80, 160, 40))
    pygame.draw.rect(screen, GRAY, (CANVAS_SIZE + 20, 140, 160, 40))
    screen.blit(pygame.font.SysFont(None, 36).render("Draw", True, BLACK), (CANVAS_SIZE + 60, 25))
    screen.blit(pygame.font.SysFont(None, 36).render("Erase", True, BLACK), (CANVAS_SIZE + 55, 85))
    screen.blit(pygame.font.SysFont(None, 36).render("Finish", True, BLACK), (CANVAS_SIZE + 50, 145))

# Function to draw on the canvas with gradient intensity
def draw_on_canvas(pos, intensity):
    global canvas
    for x in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
        for y in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
            dx, dy = pos[0] // CELL_SIZE + x, pos[1] // CELL_SIZE + y
            if 0 <= dx < GRID_SIZE and 0 <= dy < GRID_SIZE:
                dist = np.sqrt(x**2 + y**2)
                if dist <= BRUSH_RADIUS:
                    alpha = max(0, (1 - dist / BRUSH_RADIUS)**2) * intensity
                    if draw_mode:
                        canvas[dy, dx] = min(1, canvas[dy, dx] + alpha)
                    else:  # Erase mode
                        canvas[dy, dx] = max(0, canvas[dy, dx] - alpha)

# Main loop
running = True
pixel_values = None  # Store the final pixel values

while running:
    screen.fill(BLACK)
    draw_buttons()
    
    # Draw the grid based on the canvas values
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            intensity = canvas[j, i]
            color = (intensity * 255, intensity * 255, intensity * 255)
            pygame.draw.rect(screen, color, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if CANVAS_SIZE + 20 <= x <= CANVAS_SIZE + 180:
                if 20 <= y <= 60:
                    draw_mode = True
                elif 80 <= y <= 120:
                    draw_mode = False
                elif 140 <= y <= 180:
                    # Apply Gaussian blur for softer, MNIST-style strokes
                    canvas_blurred = gaussian_filter(canvas, sigma=0.75)
                    pixel_values = canvas_blurred.flatten().tolist()
                    # print("Pixel values stored:", pixel_values)
                    running = False
        elif event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
            draw_on_canvas(pygame.mouse.get_pos(), brush_intensity)

    pygame.display.flip()

pygame.quit()

# Load the saved model
def load_model(filename="trained_model.pkl"):
    with open(filename, "rb") as f:
        model_params = pickle.load(f)
    
    model = SimpleNN()
    model.weights1 = model_params["weights1"]
    model.biases1 = model_params["biases1"]
    model.weights2 = model_params["weights2"]
    model.biases2 = model_params["biases2"]
    model.weights3 = model_params["weights3"]
    model.biases3 = model_params["biases3"]
    return model

# Instantiate and load trained model
model = load_model()

# Convert pixel values to NumPy array and use as input to the model
if pixel_values is not None:
    pixel_values = np.array(pixel_values)
    output, predicted_digit = model.forward(pixel_values)
    predicted_digit = int(predicted_digit[0])  # Extract scalar from array
    confidence = softmax(output[0])[predicted_digit] * 100  # Use output[0] for correct shape
    print("Predicted digit:\33[32m\033[1m>>>", predicted_digit, "<<<\033[0m", "  Confidence:", round(confidence, 2), "%")