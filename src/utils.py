import torch


def slerp(start: torch.Tensor, end: torch.Tensor, t: float):
    omega = torch.arccos(
        torch.clip(
            torch.dot(start / torch.linalg.norm(start), end / torch.linalg.norm(end)), -1, 1
        )
    )
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - t) * start + t * end  # L'Hopital's rule/LERP
    return torch.sin((1.0 - t) * omega) / so * start + torch.sin(t * omega) / so * end
