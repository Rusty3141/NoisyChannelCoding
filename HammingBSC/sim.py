import random


def Parity(data):
    return str(data[0] ^ data[1] ^ data[2]) + str(data[1] ^ data[2] ^ data[3]) + str(data[2] ^ data[3] ^ data[0])


def Probability(received, transmitted, total, flipProb):
    result = 1

    result /= total

    for i in range(len(received)):
        if received[i] != transmitted[i]:
            result *= flipProb

    return result


dataBits = 4
maximum = 2**dataBits

probs = []

f = 0.5

source = "1010010"
received = ""

vectors = []

print(f"Original: {source}")

for i in range(len(source)):
    if random.uniform(0, 1) < f:
        received += "0" if source[i] == "1" else "0"
    else:
        received += source[i]


print(f"Received: {received}")

print(f"{maximum} transmitted codewords are possible:")

for i in range(maximum):
    data = f"{i:04b}"
    vectors.append(data + Parity([0 if x == "0" else 1 for x in data]))

    probs.append(Probability(source, vectors[i], maximum, f))

s = sum(probs)

for i, codeword in enumerate(vectors):
    print(f"{codeword} with probability {probs[i]/s}")

tol = 0.0001

probThreshold = max(probs)

print("Maximum likelihood decodes: ", end='')

for i, codeword in enumerate(vectors):
    if abs(probs[i] - probThreshold) < tol:
        print(f"{codeword[0:dataBits]} ", end='')
