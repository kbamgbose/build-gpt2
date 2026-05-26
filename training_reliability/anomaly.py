import math
from training_reliability import Warning


def check_anomaly(step, loss, grad_norm, activation_std=None):
    if math.isnan(loss) or math.isinf(loss):
        return Warning(
            step=step,
            monitor='anomaly',
            message='NaN loss detected',
            values={'loss': loss},
        )
    if math.isnan(grad_norm) or math.isinf(grad_norm):
        return Warning(
            step=step,
            monitor='anomaly',
            message='NaN grad_norm detected',
            values={'grad_norm': grad_norm},
        )
    if activation_std is not None:
        if activation_std > 50.0:
            return Warning(
                step=step,
                monitor='anomaly',
                message='activation explosion',
                values={'activation_std': activation_std},
            )
        if activation_std < 0.01 and step > 5:
            return Warning(
                step=step,
                monitor='anomaly',
                message='activation collapse',
                values={'activation_std': activation_std},
            )
    return None
