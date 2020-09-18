import torch

def UseGPU():
    '''
    gpu 사용 선언
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    return device