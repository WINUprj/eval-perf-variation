from dataclasses import dataclass


@dataclass
class Metrics:
    """Stores the key-value pair of the metrics to track in training."""
    # Agent performance-related metrics
    episode_length: str = "episode_length"
    episodic_return: str = "episodic_return"
    update_duration: str = "update_duration"

    # Critic-related metrics
    critic_loss: str = "critic_loss"
    critic_dormant_neurons: str = "critic_dormant_neurons"
    critic_layerwise_dormant_neurons: str = "critic_layerwise_dormant_neurons"
    critic_param_norm: str = "critic_param_norm"
    q1_param_norm: str = "q1_param_norm"
    q2_param_norm: str = "q2_param_norm"
    critic_grad_norm: str = "critic_grad_norm"
    q1_grad_norm: str = "q1_grad_norm"
    q2_grad_norm: str = "q2_grad_norm"
    critic_loss_diff: str = "critic_loss_diff"
    critic_grad_inner_prod_true: str = "critic_grad_inner_prod_true"
    critic_grad_inner_prod_approx: str = "critic_grad_inner_prod_approx"
    critic_update_inner_prod_true: str = "critic_update_inner_prod_true"
    critic_update_inner_prod_approx: str = "critic_update_inner_prod_approx"
    critic_update_duration: str = "critic_update_duration"
    critic_feature_srank: str = "critic_feature_srank"

    # Actor-related metrics
    actor_loss: str = "actor_loss"
    actor_dormant_neurons: str = "actor_dormant_neurons"
    actor_layerwise_dormant_neurons: str = "actor_layerwise_dormant_neurons"
    actor_param_norm: str = "actor_param_norm"
    actor_grad_norm: str = "actor_grad_norm"
    actor_loss_diff: str = "actor_loss_diff"
    entropy: str = "entropy"
    actor_grad_inner_prod_true: str = "actor_grad_inner_prod_true"
    actor_grad_inner_prod_approx: str = "actor_grad_inner_prod_approx"
    actor_update_inner_prod_true: str = "actor_update_inner_prod_true"
    actor_update_inner_prod_approx: str = "actor_update_inner_prod_approx"
    actor_update_duration: str = "actor_update_duration"
    score_func_grad_inner_prod_true: str = "score_func_grad_inner_prod_true"
    score_func_grad_inner_prod_approx: str = "score_func_grad_inner_prod_approx"
    entropy_temperature: str = "entropy_temperature"
    entropy_reg: str = "entropy_regularizer"
    actor_feature_srank: str = "actor_feature_srank"

    # Summary metrics
    binned_learning_curve: str = "binned_learning_curve"
    percentile: str = "percentile"
    percentile_stats: str = "percentile_stats"
    percentile_sorted_idx: str = "percentile_sorted_idx"
    bin_size: str = "bin_size"

    # OSP related values
    actor_instant_inner_prod: str = "actor_instant_inner_prod"
    actor_sustained_inner_prod: str = "actor_sustained_inner_prod"
    actor_sustained_counter: str = "actor_sustained_counter"
    actor_ss: str = "actor_step_size"

    critic_instant_inner_prod: str = "critic_instant_inner_prod"
    critic_sustained_inner_prod: str = "critic_sustained_inner_prod"
    critic_sustained_counter: str = "critic_sustained_counter"
    critic_ss: str = "critic_step_size"
