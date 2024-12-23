import tensorflow as tf

def main():
    # Check if TensorFlow can access a GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("Hello, GPU World! TensorFlow is using CUDA.")
        for gpu in gpus:
            print(f"Device: {gpu}")
    else:
        print("Hello, CPU World! No GPU with CUDA found.")

    # Simple TensorFlow operation
    hello = tf.constant("Hello, TensorFlow World!")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.print(hello)

if __name__ == "__main__":
    main()
