import copy
import sys
import random
from PIL import Image, ImageOps


def Parity(data):
    return [data[0] ^ data[1] ^ data[2], data[1] ^ data[2] ^ data[3], data[2] ^ data[3] ^ data[0]]


def Probability(received, transmitted, total, flipProb):
    result = 1

    for i in range(len(received)):
        if int(received[i]) != transmitted[i]:
            result *= flipProb

    return result


f = 0.1

img = Image.open(sys.argv[1]).convert("L")
img = ImageOps.exif_transpose(img)

img.save("GreyscaleOriginal.png")
print("Converted to greyscale.")

pixels = img.load()

blocks = []
decoderBlocks = []

for i in range(img.size[0]):
    for j in range(img.size[1]):
        pixelVal = [int(x) for x in list(format(pixels[i, j], f"#010b")[2:])]
        blocks.append(pixelVal[:4] + Parity(pixelVal[:4]))
        blocks.append(pixelVal[4:] + Parity(pixelVal[4:]))

print("Encoded image.")

for i in range(len(blocks)):
    for j in range(4+3):
        if random.uniform(0, 1) < f:
            blocks[i][j] = 0 if blocks[i][j] == 1 else 1
        else:
            blocks[i][j] = 1 if blocks[i][j] == 1 else 0

for i in range(0, len(blocks), 2):
    pixel = "".join([str(x) for x in blocks[i]])[:4] + \
        "".join([str(x) for x in blocks[i+1]])[:4]
    pixels[i/2 // img.size[1], i/2 % img.size[1]] = int(pixel, 2)

img.save(f"Noisy(f={f}).png")
img.show()
print("Simulated transmission by adding noise.")

for i, block in enumerate(blocks):
    probs = []
    vectors = []
    string = "".join([str(x) for x in block])
    for j in range(16):
        data = f"{j:04b}"
        vectors.append([int(x) for x in
                        data + "".join([str(x) for x in Parity([0 if x == "0" else 1 for x in data])])])

        probs.append(Probability(string, vectors[j], 4, f))

    tol = 0.0001
    probThreshold = max(probs)

    for k, codeword in enumerate(vectors):
        if abs(probs[k] - probThreshold) < tol:
            blocks[i] = vectors[k]
            next

for i in range(0, len(blocks), 2):
    pixel = "".join([str(x) for x in blocks[i]])[:4] + \
        "".join([str(x) for x in blocks[i+1]])[:4]
    pixels[i/2 // img.size[1], i/2 %
           img.size[1]] = int(pixel, 2)

img.save(f"Decoded(f={f}).png")
img.show()
print("Attempted recovery of source image.")
