import torch.optim as optim

'''
레이어별 learning rate decay를 적용 
'''
class LayerDecayOptimizer:
    '''
    __init__:
    layerwise_decay_rate: 각 레이어에 대한 감쇠율 리스트
    param_groups: 기본 optimizer에서 파라미터 그룹을 저장
                  일반적으로 각 딕셔너리가 파라미터와 하이퍼파라미터를 포함하는 딕셔러니 리스트
    '''
    def __init__(self, optimizer, layerwise_decay_rate):
        self.optimizer = optimizer
        self.layerwise_decay_rate = layerwise_decay_rate
        self.param_groups = optimizer.param_groups
    
    '''
    enumerate(self.optimizer.param_groups):
    각 파라미커 그룹과 해당 인덱스를 반복합니다.

    group['lr'] *= self.layerwise_decay_rate[i]:
    현재 파라미터 그룹의 학습률에 해당 감쇠율을 곱합니다.

    self.optimizer.step(*args, **kwargs):
    기본 옵티마이저의 'step' 메서드를 호출하여 모델 파라미터를 업데이트합니다.
    '''
    def step(self, *args, **kwargs):
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] *= self.layerwise_decay_rate[i]
        self.optimizer.step(*args, **kwargs)

    
    ''' 
    모든 최적화된 파라미터의 그래디언트를 초기화합니다.

    self.optimizer.zero_grad(*args, **kwargs):
    기본 옵티마이저의 'zero_grad' 메서드로 호출을 전달합니다.
    '''
    def zero_grad(self, *args, **kwargs):
        self.optimizer.zeor_grad(*args, **kwargs)