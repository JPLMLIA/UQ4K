pass
# # Tests with pytest for the classes and methods in objective_function.py
# #
# # Author   : Mike Stanley
# # Written  : August 26, 2021
# # Last Mod : August 26, 2021

# from objective_function import Objective
# from pred_prey import PredPrey

# # Set up a predator-prey model
# ALPHA, BETA, DELTA, GAMMA = 0.55, 0.025, 0.02, 0.8
# PREY_INIT, PRED_INIT = 30, 10
# eval_times_hr = np.linspace(start=0, stop=20, num=200)
# predprey = PredPrey(
#     alpha=ALPHA, alpha_bounds=(-5, 5),
#     gamma=GAMMA, gamma_bounds=(-5, 5),
#     beta=BETA, delta=DELTA,
#     prey_init=PREY_INIT, pred_init=PRED_INIT
# )

# # generate some data
# SIGMA = 5
# data = predprey(eval_times_hr) + stats.norm(
#     loc=0, scale=SIGMA
# ).rvs(size=(eval_times_hr.shape, 2))

# # create the objective function object
# MU = 1e13
# M_ALPHA = 10
# objective_obj = Objective(
#     forward_model=pred_prey,
#     mu=MU,
#     M_alpha=M_ALPHA
#     data=data
# )

# def test_sum_sq_norms():
#     """ TODO: update forward model to take parameters """
#     # assert objective_obj.sum_sq_norms(params=[])
#     pass

# def test_center_dist():
#     assert objective_obj.center_dist(
#         new_point=np.array([9, 8]),
#         center=np.array([4, 4])
#     ) == 41.0

# def test_call():
#     """ TODO: make this test more legit than type """
#     assert isinstance(
#         objective_obj.center_dist(
#             new_point=np.array([9, 8]),
#             center=np.array([4, 4])
#         ),
#         float
#     )
