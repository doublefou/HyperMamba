import torch
from hypermamba.model import HyperMamba_Tiny

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperMamba_Tiny(num_classes=1000).to(device)
    model.eval()
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)

if __name__ == "__main__":
    main()
