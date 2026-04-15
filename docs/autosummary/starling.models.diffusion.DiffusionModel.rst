starling.models.diffusion.DiffusionModel
========================================

.. currentmodule:: starling.models.diffusion

.. autoclass:: DiffusionModel
   :members:
   :show-inheritance:
   :special-members: __init__, __call__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~DiffusionModel.__init__
      ~DiffusionModel.add_module
      ~DiffusionModel.all_gather
      ~DiffusionModel.apply
      ~DiffusionModel.backward
      ~DiffusionModel.bfloat16
      ~DiffusionModel.buffers
      ~DiffusionModel.children
      ~DiffusionModel.clip_gradients
      ~DiffusionModel.compile
      ~DiffusionModel.compute_snr
      ~DiffusionModel.configure_callbacks
      ~DiffusionModel.configure_gradient_clipping
      ~DiffusionModel.configure_model
      ~DiffusionModel.configure_optimizers
      ~DiffusionModel.configure_sharded_model
      ~DiffusionModel.cpu
      ~DiffusionModel.cuda
      ~DiffusionModel.double
      ~DiffusionModel.eval
      ~DiffusionModel.extra_repr
      ~DiffusionModel.float
      ~DiffusionModel.forward
      ~DiffusionModel.freeze
      ~DiffusionModel.get_buffer
      ~DiffusionModel.get_extra_state
      ~DiffusionModel.get_parameter
      ~DiffusionModel.get_submodule
      ~DiffusionModel.half
      ~DiffusionModel.ipu
      ~DiffusionModel.load_from_checkpoint
      ~DiffusionModel.load_state_dict
      ~DiffusionModel.log
      ~DiffusionModel.log_dict
      ~DiffusionModel.lr_scheduler_step
      ~DiffusionModel.lr_schedulers
      ~DiffusionModel.manual_backward
      ~DiffusionModel.modules
      ~DiffusionModel.mtia
      ~DiffusionModel.named_buffers
      ~DiffusionModel.named_children
      ~DiffusionModel.named_modules
      ~DiffusionModel.named_parameters
      ~DiffusionModel.on_after_backward
      ~DiffusionModel.on_after_batch_transfer
      ~DiffusionModel.on_before_backward
      ~DiffusionModel.on_before_batch_transfer
      ~DiffusionModel.on_before_optimizer_step
      ~DiffusionModel.on_before_zero_grad
      ~DiffusionModel.on_fit_end
      ~DiffusionModel.on_fit_start
      ~DiffusionModel.on_load_checkpoint
      ~DiffusionModel.on_predict_batch_end
      ~DiffusionModel.on_predict_batch_start
      ~DiffusionModel.on_predict_end
      ~DiffusionModel.on_predict_epoch_end
      ~DiffusionModel.on_predict_epoch_start
      ~DiffusionModel.on_predict_model_eval
      ~DiffusionModel.on_predict_start
      ~DiffusionModel.on_save_checkpoint
      ~DiffusionModel.on_test_batch_end
      ~DiffusionModel.on_test_batch_start
      ~DiffusionModel.on_test_end
      ~DiffusionModel.on_test_epoch_end
      ~DiffusionModel.on_test_epoch_start
      ~DiffusionModel.on_test_model_eval
      ~DiffusionModel.on_test_model_train
      ~DiffusionModel.on_test_start
      ~DiffusionModel.on_train_batch_end
      ~DiffusionModel.on_train_batch_start
      ~DiffusionModel.on_train_end
      ~DiffusionModel.on_train_epoch_end
      ~DiffusionModel.on_train_epoch_start
      ~DiffusionModel.on_train_start
      ~DiffusionModel.on_validation_batch_end
      ~DiffusionModel.on_validation_batch_start
      ~DiffusionModel.on_validation_end
      ~DiffusionModel.on_validation_epoch_end
      ~DiffusionModel.on_validation_epoch_start
      ~DiffusionModel.on_validation_model_eval
      ~DiffusionModel.on_validation_model_train
      ~DiffusionModel.on_validation_model_zero_grad
      ~DiffusionModel.on_validation_start
      ~DiffusionModel.optimizer_step
      ~DiffusionModel.optimizer_zero_grad
      ~DiffusionModel.optimizers
      ~DiffusionModel.p_loss
      ~DiffusionModel.parameters
      ~DiffusionModel.predict_dataloader
      ~DiffusionModel.predict_step
      ~DiffusionModel.prepare_data
      ~DiffusionModel.print
      ~DiffusionModel.q_sample
      ~DiffusionModel.register_backward_hook
      ~DiffusionModel.register_buffer
      ~DiffusionModel.register_forward_hook
      ~DiffusionModel.register_forward_pre_hook
      ~DiffusionModel.register_full_backward_hook
      ~DiffusionModel.register_full_backward_pre_hook
      ~DiffusionModel.register_load_state_dict_post_hook
      ~DiffusionModel.register_load_state_dict_pre_hook
      ~DiffusionModel.register_module
      ~DiffusionModel.register_parameter
      ~DiffusionModel.register_state_dict_post_hook
      ~DiffusionModel.register_state_dict_pre_hook
      ~DiffusionModel.requires_grad_
      ~DiffusionModel.save_hyperparameters
      ~DiffusionModel.sequence2labels
      ~DiffusionModel.set_extra_state
      ~DiffusionModel.set_submodule
      ~DiffusionModel.setup
      ~DiffusionModel.share_memory
      ~DiffusionModel.state_dict
      ~DiffusionModel.teardown
      ~DiffusionModel.test_dataloader
      ~DiffusionModel.test_step
      ~DiffusionModel.to
      ~DiffusionModel.to_empty
      ~DiffusionModel.to_onnx
      ~DiffusionModel.to_torchscript
      ~DiffusionModel.toggle_optimizer
      ~DiffusionModel.toggled_optimizer
      ~DiffusionModel.train
      ~DiffusionModel.train_dataloader
      ~DiffusionModel.training_step
      ~DiffusionModel.transfer_batch_to_device
      ~DiffusionModel.type
      ~DiffusionModel.unfreeze
      ~DiffusionModel.untoggle_optimizer
      ~DiffusionModel.val_dataloader
      ~DiffusionModel.validation_step
      ~DiffusionModel.xpu
      ~DiffusionModel.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~DiffusionModel.CHECKPOINT_HYPER_PARAMS_KEY
      ~DiffusionModel.CHECKPOINT_HYPER_PARAMS_NAME
      ~DiffusionModel.CHECKPOINT_HYPER_PARAMS_TYPE
      ~DiffusionModel.SCHEDULER_MAPPING
      ~DiffusionModel.T_destination
      ~DiffusionModel.automatic_optimization
      ~DiffusionModel.call_super_init
      ~DiffusionModel.current_epoch
      ~DiffusionModel.device
      ~DiffusionModel.device_mesh
      ~DiffusionModel.dtype
      ~DiffusionModel.dump_patches
      ~DiffusionModel.example_input_array
      ~DiffusionModel.fabric
      ~DiffusionModel.global_rank
      ~DiffusionModel.global_step
      ~DiffusionModel.hparams
      ~DiffusionModel.hparams_initial
      ~DiffusionModel.local_rank
      ~DiffusionModel.logger
      ~DiffusionModel.loggers
      ~DiffusionModel.on_gpu
      ~DiffusionModel.strict_loading
      ~DiffusionModel.trainer
      ~DiffusionModel.training
   
   