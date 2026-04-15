starling.models.continuous\_diffusion.ContinuousDiffusion
=========================================================

.. currentmodule:: starling.models.continuous_diffusion

.. autoclass:: ContinuousDiffusion
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~ContinuousDiffusion.__init__
      ~ContinuousDiffusion.add_module
      ~ContinuousDiffusion.all_gather
      ~ContinuousDiffusion.apply
      ~ContinuousDiffusion.backward
      ~ContinuousDiffusion.bfloat16
      ~ContinuousDiffusion.buffers
      ~ContinuousDiffusion.children
      ~ContinuousDiffusion.clip_gradients
      ~ContinuousDiffusion.compile
      ~ContinuousDiffusion.configure_callbacks
      ~ContinuousDiffusion.configure_gradient_clipping
      ~ContinuousDiffusion.configure_model
      ~ContinuousDiffusion.configure_optimizers
      ~ContinuousDiffusion.configure_sharded_model
      ~ContinuousDiffusion.cpu
      ~ContinuousDiffusion.cuda
      ~ContinuousDiffusion.double
      ~ContinuousDiffusion.eval
      ~ContinuousDiffusion.extra_repr
      ~ContinuousDiffusion.float
      ~ContinuousDiffusion.forward
      ~ContinuousDiffusion.freeze
      ~ContinuousDiffusion.get_buffer
      ~ContinuousDiffusion.get_extra_state
      ~ContinuousDiffusion.get_parameter
      ~ContinuousDiffusion.get_submodule
      ~ContinuousDiffusion.half
      ~ContinuousDiffusion.ipu
      ~ContinuousDiffusion.load_from_checkpoint
      ~ContinuousDiffusion.load_state_dict
      ~ContinuousDiffusion.log
      ~ContinuousDiffusion.log_dict
      ~ContinuousDiffusion.lr_scheduler_step
      ~ContinuousDiffusion.lr_schedulers
      ~ContinuousDiffusion.manual_backward
      ~ContinuousDiffusion.modules
      ~ContinuousDiffusion.mtia
      ~ContinuousDiffusion.named_buffers
      ~ContinuousDiffusion.named_children
      ~ContinuousDiffusion.named_modules
      ~ContinuousDiffusion.named_parameters
      ~ContinuousDiffusion.on_after_backward
      ~ContinuousDiffusion.on_after_batch_transfer
      ~ContinuousDiffusion.on_before_backward
      ~ContinuousDiffusion.on_before_batch_transfer
      ~ContinuousDiffusion.on_before_optimizer_step
      ~ContinuousDiffusion.on_before_zero_grad
      ~ContinuousDiffusion.on_fit_end
      ~ContinuousDiffusion.on_fit_start
      ~ContinuousDiffusion.on_load_checkpoint
      ~ContinuousDiffusion.on_predict_batch_end
      ~ContinuousDiffusion.on_predict_batch_start
      ~ContinuousDiffusion.on_predict_end
      ~ContinuousDiffusion.on_predict_epoch_end
      ~ContinuousDiffusion.on_predict_epoch_start
      ~ContinuousDiffusion.on_predict_model_eval
      ~ContinuousDiffusion.on_predict_start
      ~ContinuousDiffusion.on_save_checkpoint
      ~ContinuousDiffusion.on_test_batch_end
      ~ContinuousDiffusion.on_test_batch_start
      ~ContinuousDiffusion.on_test_end
      ~ContinuousDiffusion.on_test_epoch_end
      ~ContinuousDiffusion.on_test_epoch_start
      ~ContinuousDiffusion.on_test_model_eval
      ~ContinuousDiffusion.on_test_model_train
      ~ContinuousDiffusion.on_test_start
      ~ContinuousDiffusion.on_train_batch_end
      ~ContinuousDiffusion.on_train_batch_start
      ~ContinuousDiffusion.on_train_end
      ~ContinuousDiffusion.on_train_epoch_end
      ~ContinuousDiffusion.on_train_epoch_start
      ~ContinuousDiffusion.on_train_start
      ~ContinuousDiffusion.on_validation_batch_end
      ~ContinuousDiffusion.on_validation_batch_start
      ~ContinuousDiffusion.on_validation_end
      ~ContinuousDiffusion.on_validation_epoch_end
      ~ContinuousDiffusion.on_validation_epoch_start
      ~ContinuousDiffusion.on_validation_model_eval
      ~ContinuousDiffusion.on_validation_model_train
      ~ContinuousDiffusion.on_validation_model_zero_grad
      ~ContinuousDiffusion.on_validation_start
      ~ContinuousDiffusion.optimizer_step
      ~ContinuousDiffusion.optimizer_zero_grad
      ~ContinuousDiffusion.optimizers
      ~ContinuousDiffusion.p_losses
      ~ContinuousDiffusion.parameters
      ~ContinuousDiffusion.predict_dataloader
      ~ContinuousDiffusion.predict_step
      ~ContinuousDiffusion.prepare_data
      ~ContinuousDiffusion.print
      ~ContinuousDiffusion.q_sample
      ~ContinuousDiffusion.random_times
      ~ContinuousDiffusion.register_backward_hook
      ~ContinuousDiffusion.register_buffer
      ~ContinuousDiffusion.register_forward_hook
      ~ContinuousDiffusion.register_forward_pre_hook
      ~ContinuousDiffusion.register_full_backward_hook
      ~ContinuousDiffusion.register_full_backward_pre_hook
      ~ContinuousDiffusion.register_load_state_dict_post_hook
      ~ContinuousDiffusion.register_load_state_dict_pre_hook
      ~ContinuousDiffusion.register_module
      ~ContinuousDiffusion.register_parameter
      ~ContinuousDiffusion.register_state_dict_post_hook
      ~ContinuousDiffusion.register_state_dict_pre_hook
      ~ContinuousDiffusion.requires_grad_
      ~ContinuousDiffusion.save_hyperparameters
      ~ContinuousDiffusion.sequence2labels
      ~ContinuousDiffusion.set_extra_state
      ~ContinuousDiffusion.set_submodule
      ~ContinuousDiffusion.setup
      ~ContinuousDiffusion.share_memory
      ~ContinuousDiffusion.state_dict
      ~ContinuousDiffusion.teardown
      ~ContinuousDiffusion.test_dataloader
      ~ContinuousDiffusion.test_step
      ~ContinuousDiffusion.to
      ~ContinuousDiffusion.to_empty
      ~ContinuousDiffusion.to_onnx
      ~ContinuousDiffusion.to_torchscript
      ~ContinuousDiffusion.toggle_optimizer
      ~ContinuousDiffusion.toggled_optimizer
      ~ContinuousDiffusion.train
      ~ContinuousDiffusion.train_dataloader
      ~ContinuousDiffusion.training_step
      ~ContinuousDiffusion.transfer_batch_to_device
      ~ContinuousDiffusion.type
      ~ContinuousDiffusion.unfreeze
      ~ContinuousDiffusion.untoggle_optimizer
      ~ContinuousDiffusion.val_dataloader
      ~ContinuousDiffusion.validation_step
      ~ContinuousDiffusion.xpu
      ~ContinuousDiffusion.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~ContinuousDiffusion.CHECKPOINT_HYPER_PARAMS_KEY
      ~ContinuousDiffusion.CHECKPOINT_HYPER_PARAMS_NAME
      ~ContinuousDiffusion.CHECKPOINT_HYPER_PARAMS_TYPE
      ~ContinuousDiffusion.T_destination
      ~ContinuousDiffusion.automatic_optimization
      ~ContinuousDiffusion.call_super_init
      ~ContinuousDiffusion.current_epoch
      ~ContinuousDiffusion.device
      ~ContinuousDiffusion.device_mesh
      ~ContinuousDiffusion.dtype
      ~ContinuousDiffusion.dump_patches
      ~ContinuousDiffusion.example_input_array
      ~ContinuousDiffusion.fabric
      ~ContinuousDiffusion.global_rank
      ~ContinuousDiffusion.global_step
      ~ContinuousDiffusion.hparams
      ~ContinuousDiffusion.hparams_initial
      ~ContinuousDiffusion.local_rank
      ~ContinuousDiffusion.logger
      ~ContinuousDiffusion.loggers
      ~ContinuousDiffusion.on_gpu
      ~ContinuousDiffusion.strict_loading
      ~ContinuousDiffusion.trainer
      ~ContinuousDiffusion.training
   
   