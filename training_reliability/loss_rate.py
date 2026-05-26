from training_reliability import Warning


def check_loss_rate(step, loss_history):
    if len(loss_history) >= 11:
        rate = loss_history[-1] / loss_history[-11]
        if rate > 1.5:
            return Warning(
                step=step,
                monitor='loss_rate',
                message='loss spike',
                values={'rate': rate, 'current_loss': loss_history[-1], 'prev_loss': loss_history[-11]},
            )

    if len(loss_history) >= 20:
        rate = loss_history[-1] / loss_history[-20]
        if rate > 0.99:
            return Warning(
                step=step,
                monitor='loss_rate',
                message='training stalled',
                values={'rate': rate, 'current_loss': loss_history[-1], 'prev_loss': loss_history[-20]},
            )

    return None
