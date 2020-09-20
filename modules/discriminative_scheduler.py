from overrides import overrides
import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler \
    import LearningRateScheduler

@LearningRateScheduler.register('discriminative')
class DiscriminativeLR(LearningRateScheduler):
    """
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 last_epoch: int = -1,
                 discriminative_fine_tuning: bool = False,
                 gradual_unfreezing: bool = False,
                 decay_factor: float = 0.38) -> None:
        assert discriminative_fine_tuning or gradual_unfreezing, \
            "At least one of discriminative fine tuning or gradual " \
            "unfreezing should be enabled"
        if self.gradual_unfreezing:
            assert not optimizer.param_groups[-1]["params"], \
                "The default group should be empty."
        if self.gradual_unfreezing or discriminative_fine_tuning:
            assert len(optimizer.param_groups) > 2, \
                "There should be at least 3 param_groups (2 + empty default group)" \
                " for gradual unfreezing / discriminative fine-tuning to make sense."
                
        super().__init__(optimizer, last_epoch=last_epoch)
        self.gradual_unfreezing = gradual_unfreezing
        self.freezing_current = self.gradual_unfreezing
        self.is_first_epoch = True
        if discriminative_fine_tuning:
            # TODO: see if there's a way to do discriminative fine-tuning
            # by named param group
            # skip the last param_group if it has no parameters
            exponent = 0
            for i in range(len(self.base_values) - 1, -1, -1):
                param_group = optimizer.param_groups[i]
                if param_group['params']:
                    param_group['lr'] = \
                        self.base_values[i] * decay_factor ** exponent
                    self.base_values[i] = param_group['lr']
                    exponent += 1
    
    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        if self.gradual_unfreezing:
            # the method is called once when initialising before the
            # first epoch (epoch -1) and then always at the end of each
            # epoch; so the first time, with epoch id -1, we want to set
            # up for epoch #1; the second time, with epoch id 0,
            # we want to set up for epoch #2, etc.
            if self.is_first_epoch:
                num_layers_to_unfreeze = 1
                self.is_first_epoch = False
            else:
                num_layers_to_unfreeze = epoch + 2
            if num_layers_to_unfreeze >= len(self.optimizer.param_groups)-1:
                logger.info('Gradual unfreezing finished. Training all layers.')
                self.freezing_current = False
            else:
                logger.info(f'Gradual unfreezing. Training only the top {num_layers_to_unfreeze} layers.')
            for i, param_group in enumerate(reversed(self.optimizer.param_groups)):
                for param in param_group["params"]:
                    # i = 0 is the default group; we care about i > 0
                    param.requires_grad = bool(i <= num_layers_to_unfreeze)

    def get_values(self):
        return self.base_values