#idk why but number of iterations for this code to find two equal numbers is always 911708
import torch

torch.manual_seed(0)

rta = torch.rand(3, 4)
rtb = torch.rand(3, 4)

print("Initial Tensors:")
print("rta:")
print(rta)
print("rtb:")
print(rtb)

counter = 0

while not torch.any(torch.eq(rta.unsqueeze(0), rtb.unsqueeze(1))):
    rta = torch.rand(3, 4)
    rtb = torch.rand(3, 4)
    counter += 1

print("\nMatching Tensors:")
print("rta:")
print(rta)
print("rtb:")
print(rtb)

matching_elements = torch.any(torch.eq(rta.unsqueeze(0), rtb.unsqueeze(1)))

print("\nIs there at least one matching element?")
print(matching_elements)

print(f"\nNumber of iterations to find at least one matching element: {counter}")
