# Copyright (c) 2017- Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import inspect
import logging
from enum import Enum
from collections import defaultdict

import tqdm
import torch

import onegan
import onegan.loss as losses
from onegan.option import AttrDict
from onegan.extension import History, TensorBoardLogger, GANCheckpoint


class Events(Enum):
    """ Events for the estimator  """
    STARTED = 'started'
    END = 'end'
    ITERATION_START = 'iteration_start'
    ITERATION_END = 'iteration_end'
    EPOCH_START = 'epoch_start'
    EPOCH_END = 'epoch_end'


class EstimatorEventMixin:
    """ Mixin for the event-triggered estimator.

    Maim implementation comes from `https://github.com/pytorch/ignite/blob/master/ignite/engine/engine.py`
    """

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        """ Add an event handler to be executed when the specified event is triggered.

        Args:
            event_name (Events): event the handler attach to
            handler (Callable): the callable function that should be invoked
            *args: optional args to be passed to `handler`
            **kwargs: optional keyword args to be passed to `handler`

        Notes:
            The handler function's first argument will be `self` (the `Estimator`).

        Examples:

            >>> def print_epoch(estimator):
            >>>    print("Epoch: {}".format(estimator.state.epoch))
            >>> estimator.add_event_handler(Events.EPOCH_END, print_epoch)
        """
        if event_name not in Events:
            self._log.error(f'attempt to add event handler to an invalid event {event_name}')
            raise ValueError(f'Event {event_name} is not a valid event')

        self._check_signature(handler, 'handler', *args, **kwargs)
        self._events[event_name].append((handler, args, kwargs))
        self._log.debug(f'Handler added for event {event_name}')

    def on(self, event_name, *args, **kwargs):
        """ Decorator shortcut for add_event_handler.

        Args:
            event_name (Events): event the handler attach to
            *args: optional args to be passed to `handler`
            **kwargs: optional keyword args to be passed to `handler`
        """
        def decorator(f):
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f
        return decorator

    def _check_signature(self, fn, fn_description, *args, **kwargs):
        exception_msg = None

        signature = inspect.signature(fn)
        try:
            signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            fn_params = list(signature.parameters)
            exception_msg = str(exc)

        if exception_msg:
            passed_params = [self] + list(args) + list(kwargs)
            raise ValueError(f'Error adding {fn} "{fn_description}": '
                             f'takes parameters {fn_params} but will be called with {passed_params} '
                             f'({exception_msg})')

    def _trigger(self, event_name, *args):
        self._log.debug(f'trigger handlers for event {event_name}')
        for handle in self._events[event_name]:
            evt_handler, evt_args, evt_kwargs = handle
            evt_handler(self, *(args + evt_args), **evt_kwargs)


class Estimator:
    """ Base estimator for functional support. """

    def load_checkpoint(self, weight_path, remove_module=False, resume=False) -> None:
        """ load checkpoint if internal ``saver`` is not `None`. """
        if not hasattr(self, 'saver') or self.saver is None:
            return
        self.saver.load(weight_path, self.model, remove_module=remove_module, resume=resume)

    def save_checkpoint(self, save_optim=False) -> None:
        """ save checkpoint if internal ``saver`` is not `None`. """
        if not hasattr(self, 'saver') or self.saver is None:
            return
        optim = self.optimizer if save_optim else None
        self.saver.save(self.model, optim, self.state.epoch + 1)

    def adjust_learning_rate(self, monitor_val) -> None:
        """ adjust the learning rate if internal ``lr_scheduler`` is not `None`. """
        if not hasattr(self, 'lr_scheduler') or self.lr_scheduler is None:
            return
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(monitor_val)
        else:
            self.lr_scheduler.step()


'''
Event-trigger estimator
'''


def epoch_end_logging(estmt):
    estmt.tensorboard_epoch_logging(scalar=estmt.history.metric)


def iteration_end_logging(estmt):
    summary = estmt.state.get('summary', {})
    prefix, image, histogram = summary.get('prefix'), summary.get('image'), summary.get('histogram')
    estmt.tensorboard_logging(image=image, prefix=prefix)
    estmt.tensorboard_logging(histogram=histogram, prefix=prefix)


def adjust_learning_rate(estmt):
    estmt.adjust_learning_rate(estmt.history.get('loss/loss_val'))


def save_checkpoint(estmt):
    estmt.save_checkpoint()


class OneEstimator(EstimatorEventMixin, Estimator):
    r""" Estimator for network training and evaluation.

    Args:
        model (torch.nn.Module): defined model for estimator.
        optimizer (torch.optim, optional): optimizer for model training.
        lr_scheduler (torch.optim.lr_scheduler, optional): learning rate scheduler for
            model training.
        logger (extension.TensorBoardLogger, optional): training state logger (default: None).
        saver (extension.Checkpoint, optional): checkpoint persistence (default: None).
        default_handlers (bool): turn on/off the defalt handlers (default: False).

    Attributes:
        history (extension.History): internal statistics of training state.
    """

    def __init__(self, model, optimizer=None, lr_scheduler=None, logger=None, saver=None, default_handlers=False):
        self.model = model

        # can leave empty
        self.optimizer = optimizer

        # optional
        self.lr_scheduler = lr_scheduler
        self.saver = saver
        self.logger = logger

        # internal
        self.history = History()
        self.state = AttrDict(epoch=0)
        self._events = defaultdict(list)
        self._hist_dict = defaultdict(list)
        self._log = logging.getLogger('onegan.OneEstimator')

        if default_handlers:
            self.add_default_event_handlers()
        self._log.info(f'OneEstimator is initialized')

    def add_default_event_handlers(self):
        self.add_event_handler(Events.ITERATION_END, iteration_end_logging)
        self.add_event_handler(Events.EPOCH_END, epoch_end_logging)
        self.add_event_handler(Events.EPOCH_END, save_checkpoint)
        self.add_event_handler(Events.EPOCH_END, adjust_learning_rate)

    def tensorboard_logging(self, image=None, histogram=None, prefix=None):
        ''' wrapper in estimator for Tensorboard logger.

        Args:
            image: dict() of a list of images
            histogram: dict() of tensors for accumulated histogram
            prefix: prefix string for keyword-image
        '''
        if not hasattr(self, 'logger') or self.logger is None:
            return

        if image and prefix:
            self.logger.image(image, self.state.epoch, prefix)
            self._log.debug('tensorboard_logging logs images')

        if histogram and prefix:
            for tag, tensor in histogram.items():
                self._hist_dict[f'{prefix}{tag}'].append(tensor.clone())
            self._log.debug('tensorboard_logging accumulate histograms')

    def tensorboard_epoch_logging(self, scalar=None):
        ''' wrapper in estimator for Tensorboard logger.

        Args:
            scalar: dict() of a list of scalars
        '''
        if not hasattr(self, 'logger') or self.logger is None:
            return

        self.logger.scalar(scalar, self.state.epoch)
        self._log.debug('tensorboard_epoch_logging logs scalars')

        if self._hist_dict:
            kw_histograms = {tag: torch.cat(tensors) for tag, tensors in self._hist_dict.items()}
            self.logger.histogram(kw_histograms, self.state.epoch)
            self._hist_dict = defaultdict(list)
            self._log.debug('tensorboard_epoch_logging logs histograms')

    def run(self, train_loader, validate_loader, closure_fn, epochs, longtime_pbar=False):
        epoch_range = tqdm.trange(epochs, desc='Training Procedure') if longtime_pbar else range(epochs)

        for epoch in epoch_range:
            self.history.clear()
            self.state.epoch = epoch
            self._trigger(Events.EPOCH_START)

            self.train(train_loader, closure_fn, longtime_pbar)
            self.evaluate(validate_loader, closure_fn, longtime_pbar)

            self._trigger(Events.EPOCH_END)
            self._log.debug(f'OneEstimator epoch#{epoch} end')

    def train(self, data_loader, update_fn, longtime_pbar=False):
        self.model.train()
        progress = tqdm.tqdm(data_loader, desc=f'Epoch#{self.state.epoch + 1}', leave=not longtime_pbar)

        for data in progress:
            self._trigger(Events.ITERATION_START)

            result = update_fn(self.model, data)
            # `loss`, `status` should be in result (dict)

            loss = result.pop('loss')
            assert loss, 'Returned result from closure must contain key `loss` to backward()'
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            status = result.pop('status')
            assert status, 'Returned result from closure must contain key `status` for history'
            current_status = self.history.update(status)

            progress.set_postfix(current_status)
            self.state.update(result)

            self._trigger(Events.ITERATION_END)

    def evaluate(self, data_loader, inference_fn, longtime_pbar=False):
        self.model.eval()
        progress = tqdm.tqdm(data_loader, desc='evaluating', leave=not longtime_pbar)

        with torch.no_grad():
            for data in progress:
                self._trigger(Events.ITERATION_START)

                result = inference_fn(self.model, data)
                # `status` should be in result (dict)

                status = result.pop('status')
                assert status, 'Returned result from closure must contain key `status` for history'
                current_status = self.history.update(status, log_suffix='_val')

                progress.set_postfix(current_status)
                self.state.update(result)

                self._trigger(Events.ITERATION_END)


# deprecated
class OneGANEstimator:

    def __init__(self, model, optimizer=None, lr_scheduler=None, logger=None, saver=None, name=None):
        self.models = model
        self.schedulers = lr_scheduler
        self.model_g, self.model_d = model if len(model) == 2 else (None, None)
        self.optim_g, self.optim_d = optimizer if optimizer else (None, None)
        self.saver = saver
        self.logger = logger
        self.sched_g, self.sched_d = lr_scheduler if len(lr_scheduler) == 2 else (None, None)
        self.name = name

        self.history = History()
        self.history_val = History()
        self.state = {}
        self.log = logging.getLogger(f'OneGAN.{name}')
        self.log.info(f'OneGANEstimator<{name}> is initialized')

    def run(self, train_loader, validate_loader, update_fn, inference_fn, epochs):
        for epoch in range(epochs):
            self.state.epoch = epoch

            self.train(train_loader, update_fn)
            self.logger.scalar(self.history.metric, epoch)

            self.evaluate(validate_loader, inference_fn)
            self.logger.scalar(self.history.metric, epoch)

            self.save_checkpoint()
            self.adjust_learning_rate(('loss/loss_g_val', 'loss/loss_d_val'))
            self.log.debug(f'OneEstimator<{self.name}> epoch#{epoch} end')

    def load_checkpoint(self, weight_path, resume=False):
        if not hasattr(self, 'saver') or self.saver is None:
            return
        self.saver.load(self, weight_path, resume)

    def save_checkpoint(self):
        if not hasattr(self, 'saver') or self.saver is None:
            return
        self.saver.save(self, self.state.epoch)

    def adjust_learning_rate(self, monitor_vals):
        if not hasattr(self, 'lr_scheduler') or self.lr_scheduler is None:
            return
        try:
            for sched, monitor_val in zip(self.schedulers, monitor_vals):
                sched.step(self.history[monitor_val])
        except Exception:
            for sched in self.schedulers:
                sched.step()

    def train(self, data_loader, update_fn):
        self.model_g.train()
        self.model_d.train()
        self.history.clear()

        progress = tqdm.tqdm(data_loader)
        progress.set_description(f'Epoch#{self.state["epoch"] + 1}')

        for data in progress:
            staged_closure = update_fn(self.model_g, self.model_d, data)

            self.optim_d.zero_grad()
            loss_d = next(staged_closure)
            loss_d['loss/loss_d'].backward()
            self.optim_d.step()

            self.optim_g.zero_grad()
            loss_g = next(staged_closure)
            loss_g['loss/loss_g'].backward()
            self.optim_g.step()

            accuracy = next(staged_closure)
            progress.set_postfix(self.history.add({**loss_d, **loss_g, **accuracy}))
            next(staged_closure)
        return self.history.metric()

    def evaluate(self, data_loader, inference_fn):
        self.model_g.eval()
        self.model_d.eval()
        self.history.clear()

        progress = tqdm.tqdm(data_loader)
        progress.set_description('Evaluate')

        with torch.no_grad():
            for data in progress:
                staged_closure = inference_fn(self.model_g, self.model_d, data)
                loss_d, loss_g, accuracy, _ = [r for r in staged_closure]

                progress.set_postfix(self.history.add({**loss_d, **loss_g, **accuracy}, log_suffix='_val'))
            return self.history.metric()

    def dummy_run(self, train_loader, validate_loader, update_fn, inference_fn, epoch_fn, epochs):
        for epoch in range(epochs):
            self.state.epoch = epoch
            self.dummy_train(train_loader, update_fn)
            self.dummy_evaluate(validate_loader, inference_fn)
            epoch_fn(epoch)
            self.log.debug(f'OneEstimator<{self.name}> epoch#{epoch} end')

    def dummy_train(self, data_loader, update_fn):
        [m.train() for m in self.models]
        self.history.clear()

        progress = tqdm.tqdm(data_loader)
        progress.set_description(f'Epoch#{self.state["epoch"] + 1}')

        for data in progress:
            stat = {}
            for staged_closure in update_fn(self.models, data):
                if isinstance(staged_closure, tuple):
                    loss, (optim, key_loss) = staged_closure
                    optim.zero_grad()
                    loss[key_loss].backward()
                    optim.step()
                    stat.update(loss)
                elif isinstance(staged_closure, dict):
                    accuracy = staged_closure
                    stat.update(accuracy)
            progress.set_postfix(self.history.add(stat))

    def dummy_evaluate(self, data_loader, update_fn):
        [m.eval() for m in self.models]
        self.history_val.clear()

        progress = tqdm.tqdm(data_loader)
        progress.set_description(f'Epoch#{self.state["epoch"] + 1}')

        with torch.no_grad():
            for data in progress:
                stat = {}
                for staged_closure in update_fn(self.models, data):
                    if isinstance(staged_closure, tuple):
                        loss, _ = staged_closure
                        stat.update(loss)
                    elif isinstance(staged_closure, dict):
                        accuracy = staged_closure
                        stat.update(accuracy)
                progress.set_postfix(self.history_val.add(stat, log_suffix='_val'))


# deprecated
class OneGANReadyEstimator(Estimator):

    def __init__(self, model, optimizer, metric, name, **kwargs):
        self.gnet, self.dnet = model
        self.g_optim, self.d_optim = optimizer
        self.metric = metric
        self.name = name

        self.saver = GANCheckpoint(name=name, save_interval=kwargs.get('save_epochs', 5))
        self.logger = TensorBoardLogger(name=name, max_num_images=kwargs.get('max_num_images', 30))

        self.build_criterion()

    def build_criterion(self):
        self.recon_weight = 10
        self.recon_loss = losses.l1_loss
        self.adv_loss = losses.adversarial_ls_loss

    def forward_d(self, source, output, target):
        fake = losses.conditional_input(source, output, self.conditional)
        real = losses.conditional_input(source, target, self.conditional)
        d_fake = self.adv_loss(self.dnet(fake), False)
        d_real = self.adv_loss(self.dnet(real), True)
        d_terms = {'d/real': d_real, 'd/fake': d_fake, 'd/loss': d_real + d_fake}
        d_loss = d_terms['d/loss']
        return d_loss, d_terms

    def forward_g(self, source, output, target):
        recon = self.recon_loss(output, target)
        adv = self.adv_loss(self.dnet(losses.conditional_input(source, output, self.conditional)), True)
        g_terms = {'g/recon': recon, 'g/adv': adv, 'g/loss': recon * self.recon_weight + adv}
        g_loss = g_terms['g/loss']
        return g_loss, g_terms

    def run(self, train_loader, validate_loader, epochs):
        for epoch in range(epochs):
            self.gnet.train()
            self.dnet.train()
            self.train(train_loader, epoch, History())
            self.gnet.eval()
            self.dnet.eval()
            self.evaluate(validate_loader, epoch, History())
            self.save_checkpoint(epoch)

    def train(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader)
        for i, (source, target) in enumerate(progress):
            source, target = source.to(onegan.device()), target.to(onegan.device())
            output = self.gnet(source)

            d_loss, d_terms = self.forward_d(source, output.detach(), target)
            g_loss, g_terms = self.forward_g(source, output, target)
            acc = self.metric(output, target)

            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            progress.set_description('Epoch#%d' % (epoch + 1))
            progress.set_postfix(history.add({**g_terms, **d_terms, 'acc/psnr': acc}))

            self.logger.image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='train_')

        self.logger.scalar(history.metric(), epoch)

    def evaluate(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader, leave=False)

        with torch.no_grad():
            for i, (source, target) in enumerate(progress):
                source, target = source.to(onegan.device()), target.to(onegan.device())
                output = self.gnet(source)

                _, d_terms = self.forward_d(source, output.detach(), target)
                _, g_terms = self.forward_g(source, output, target)
                acc = self.metric(output, target)

                progress.set_description('Evaluate')
                progress.set_postfix(history.add({**g_terms, **d_terms, 'acc/psnr': acc}, log_suffix='_val'))

                self.logger.image(
                    {'input': source.data, 'output': output.data, 'target': target.data},
                    epoch=epoch, prefix='val_')

        self.logger.scalar(history.metric(), epoch)


# deprecated
class OneWGANReadyEstimator(OneGANEstimator):

    def __init__(self, model, optimizer, metric, saver, name):
        super().__init__(model, optimizer)
        self.critic_iters = 5

    def build_criterion(self):
        self.recon_weight = 10
        self.gp_weight = 10
        self.recon_loss = losses.l1_loss
        self.adv_loss = losses.adversarial_w_loss

    def forward_d(self, source, output, target):
        fake = losses.conditional_input(source, output, self.conditional)
        real = losses.conditional_input(source, target, self.conditional)
        d_fake = self.adv_loss(self.dnet(fake), False)
        d_real = self.adv_loss(self.dnet(real), True)
        d_gp = losses.gradient_penalty(self.dnet, real, fake)
        d_terms = {'d/real': d_real, 'd/fake': d_fake, 'd/gp': d_gp, 'd/loss': d_real + d_fake + self.gp_weight * d_gp}
        d_loss = d_terms['d/loss']
        return d_loss, d_terms

    def train(self, data_loader, epoch, history, **kwargs):
        progress = tqdm.tqdm(data_loader)

        def fetch_data():
            source, target = next(progress)
            return source.to(onegan.device()), target.to(onegan.device())

        # TODO: need go through and modify to work
        for _ in range(len(progress)):

            for _ in range(self.critic_iters):
                source, target = fetch_data()
                output = self.gnet(source).detach()
                d_loss, d_terms = self.forward_d(source, output, target)

                self.d_optim.zero_grad()
                d_loss.backward()
                self.d_optim.step()

            source, target = fetch_data()
            output = self.gnet(source)
            g_loss, g_terms = self.forward_g(source, output, target)

            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            acc = self.metric(output, target)
            progress.set_description('Epoch#%d' % (epoch + 1))
            progress.set_postfix(history.add({**g_terms, **d_terms, 'acc/psnr': acc}))

            self.logger.image(
                {'input': source.data, 'output': output.data, 'target': target.data},
                epoch=epoch, prefix='train_')

        self.logger.scalar(history.metric(), epoch)
