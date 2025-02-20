import sys

def binary_to_c_array(input_file, output_file):
    with open(input_file, 'rb') as f:
        data = f.read()

    with open(output_file, 'w') as f:
        f.write(f"unsigned char {input_file.replace('.', '_')}[] = {{\n")
        for i in range(0, len(data), 12):
            f.write("  " + ", ".join(f"0x{byte:02x}" for byte in data[i:i+12]) + ",\n")
        f.write("};\n")
        f.write(f"unsigned int {input_file.replace('.', '_')}_len = {len(data)};\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python binary_to_c_array.py <input_file> <output_file>")
    else:
        binary_to_c_array(sys.argv[1], sys.argv[2])