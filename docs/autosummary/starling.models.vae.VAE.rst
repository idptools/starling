starling.models.vae.VAE
=======================

.. currentmodule:: starling.models.vae

.. autoclass:: VAE
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~VAE.__init__
      ~VAE.add_module
      ~VAE.all_gather
      ~VAE.apply
      ~VAE.backward
      ~VAE.bfloat16
      ~VAE.buffers
      ~VAE.children
      ~VAE.clip_gradients
      ~VAE.compile
      ~VAE.configure_callbacks
      ~VAE.configure_gradient_clipping
      ~VAE.configure_model
      ~VAE.configure_optimizers
      ~VAE.configure_sharded_model
      ~VAE.cpu
      ~VAE.cuda
      ~VAE.decode
      ~VAE.double
      ~VAE.encode
      ~VAE.eval
      ~VAE.extra_repr
      ~VAE.float
      ~VAE.forward
      ~VAE.freeze
      ~VAE.gaussian_likelihood
      ~VAE.get_buffer
      ~VAE.get_extra_state
      ~VAE.get_parameter
      ~VAE.get_submodule
      ~VAE.half
      ~VAE.ipu
      ~VAE.load_from_checkpoint
      ~VAE.load_state_dict
      ~VAE.log
      ~VAE.log_dict
      ~VAE.lr_scheduler_step
      ~VAE.lr_schedulers
      ~VAE.manual_backward
      ~VAE.modules
      ~VAE.mtia
      ~VAE.named_buffers
      ~VAE.named_children
      ~VAE.named_modules
      ~VAE.named_parameters
      ~VAE.on_after_backward
      ~VAE.on_after_batch_transfer
      ~VAE.on_before_backward
      ~VAE.on_before_batch_transfer
      ~VAE.on_before_optimizer_step
      ~VAE.on_before_zero_grad
      ~VAE.on_fit_end
      ~VAE.on_fit_start
      ~VAE.on_load_checkpoint
      ~VAE.on_predict_batch_end
      ~VAE.on_predict_batch_start
      ~VAE.on_predict_end
      ~VAE.on_predict_epoch_end
      ~VAE.on_predict_epoch_start
      ~VAE.on_predict_model_eval
      ~VAE.on_predict_start
      ~VAE.on_save_checkpoint
      ~VAE.on_test_batch_end
      ~VAE.on_test_batch_start
      ~VAE.on_test_end
      ~VAE.on_test_epoch_end
      ~VAE.on_test_epoch_start
      ~VAE.on_test_model_eval
      ~VAE.on_test_model_train
      ~VAE.on_test_start
      ~VAE.on_train_batch_end
      ~VAE.on_train_batch_start
      ~VAE.on_train_end
      ~VAE.on_train_epoch_end
      ~VAE.on_train_epoch_start
      ~VAE.on_train_start
      ~VAE.on_validation_batch_end
      ~VAE.on_validation_batch_start
      ~VAE.on_validation_end
      ~VAE.on_validation_epoch_end
      ~VAE.on_validation_epoch_start
      ~VAE.on_validation_model_eval
      ~VAE.on_validation_model_train
      ~VAE.on_validation_model_zero_grad
      ~VAE.on_validation_start
      ~VAE.optimizer_step
      ~VAE.optimizer_zero_grad
      ~VAE.optimizers
      ~VAE.parameters
      ~VAE.predict_dataloader
      ~VAE.predict_step
      ~VAE.prepare_data
      ~VAE.print
      ~VAE.register_backward_hook
      ~VAE.register_buffer
      ~VAE.register_forward_hook
      ~VAE.register_forward_pre_hook
      ~VAE.register_full_backward_hook
      ~VAE.register_full_backward_pre_hook
      ~VAE.register_load_state_dict_post_hook
      ~VAE.register_load_state_dict_pre_hook
      ~VAE.register_module
      ~VAE.register_parameter
      ~VAE.register_state_dict_post_hook
      ~VAE.register_state_dict_pre_hook
      ~VAE.reparameterize
      ~VAE.requires_grad_
      ~VAE.save_hyperparameters
      ~VAE.set_extra_state
      ~VAE.set_submodule
      ~VAE.setup
      ~VAE.share_memory
      ~VAE.state_dict
      ~VAE.symmetrize
      ~VAE.teardown
      ~VAE.test_dataloader
      ~VAE.test_step
      ~VAE.to
      ~VAE.to_empty
      ~VAE.to_onnx
      ~VAE.to_torchscript
      ~VAE.toggle_optimizer
      ~VAE.toggled_optimizer
      ~VAE.train
      ~VAE.train_dataloader
      ~VAE.training_step
      ~VAE.transfer_batch_to_device
      ~VAE.type
      ~VAE.unfreeze
      ~VAE.untoggle_optimizer
      ~VAE.vae_loss
      ~VAE.val_dataloader
      ~VAE.validation_step
      ~VAE.xpu
      ~VAE.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~VAE.CHECKPOINT_HYPER_PARAMS_KEY
      ~VAE.CHECKPOINT_HYPER_PARAMS_NAME
      ~VAE.CHECKPOINT_HYPER_PARAMS_TYPE
      ~VAE.T_destination
      ~VAE.automatic_optimization
      ~VAE.call_super_init
      ~VAE.current_epoch
      ~VAE.device
      ~VAE.device_mesh
      ~VAE.dtype
      ~VAE.dump_patches
      ~VAE.example_input_array
      ~VAE.fabric
      ~VAE.global_rank
      ~VAE.global_step
      ~VAE.hparams
      ~VAE.hparams_initial
      ~VAE.local_rank
      ~VAE.logger
      ~VAE.loggers
      ~VAE.on_gpu
      ~VAE.strict_loading
      ~VAE.trainer
      ~VAE.training
   
   