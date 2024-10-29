import numpy as np
import copy


class subgroup_func:
    def __init__(self, lower_bound, upper_bound, num_subgroup):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_subgroup = num_subgroup
        self.subgroup_size = (upper_bound - lower_bound) / num_subgroup

    def get_subgroup(self, value):
        group_id = ((value - self.lower_bound) / self.subgroup_size).astype(int)
        return np.clip(group_id, 0, self.num_subgroup - 1)

    def get_subgroup_w_range(self, value, lower_bound, upper_bound):
        subgroup_size = (upper_bound - lower_bound) / self.num_subgroup
        group_id = ((value - lower_bound) / subgroup_size).astype(int)
        return np.clip(group_id, 0, self.num_subgroup - 1)


class online_gd:
    def __init__(
        self,
        path_length=40,
        num_bins=100,
        lower_bound=0,
        upper_bound=10,
        lr=1e-1,
    ):
        self.lr = lr
        self.num_bins = num_bins
        self.subgroup_func = subgroup_func(lower_bound, upper_bound, num_bins)
        self.d = np.zeros((6, path_length, num_bins)).astype(np.float32)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def update_d(self, info_gain_feedbacks, info_gains_preds, info_gain_hat):
        """
        info_gain_feedbacks: 6 (r, g, b, depth, sem, occ) x (path_length x h x w)
        info_gains_preds: 6 (r, g, b, depth, sem, occ) x (path_length x h x w)
        """
        for i in range(5):
            info_gain_feedbacks[i] = np.clip(
                info_gain_feedbacks[i], self.lower_bound, self.upper_bound
            )
            info_gains_preds[i] = np.clip(
                info_gains_preds[i], self.lower_bound, self.upper_bound
            )
            info_gain_hat[i] = np.clip(
                info_gain_hat[i], self.lower_bound, self.upper_bound
            )

        info_gain_feedbacks[5] = np.clip(info_gain_feedbacks[5], 0, 1)
        info_gains_preds[5] = np.clip(info_gains_preds[5], 0, 1)
        info_gain_hat[5] = np.clip(info_gain_hat[5], 0, 1)

        # curr_d = copy.deepcopy(self.d)

        # actual_d = info_gain_feedbacks - info_gains_preds

        # d_diff = actual_d - curr_d

        for i in range(self.d.shape[0]):
            for j in range(self.d.shape[1]):
                if i == 5:
                    subgroup_id = self.subgroup_func.get_subgroup_w_range(
                        info_gains_preds[i][j], 0, 1
                    )
                    upper_bound = 1
                    lower_bound = 0
                else:
                    subgroup_id = self.subgroup_func.get_subgroup(
                        info_gains_preds[i][j]
                    )  # h x w
                    upper_bound = self.upper_bound
                    lower_bound = self.lower_bound

                for k in range(self.d.shape[2]):
                    if np.sum(subgroup_id == k) == 0:
                        continue

                    # d_diff_k = np.sum(d_diff[i][j][subgroup_id == k])
                    actual_g_k = np.mean(info_gain_feedbacks[i][j][subgroup_id == k])
                    improved_g_k = np.mean(info_gain_hat[i][j][subgroup_id == k])

                    curr_gradient = 1 if actual_g_k > improved_g_k else -1

                    self.d[i][j][k] = self.d[i][j][k] + curr_gradient * self.lr / 2

    def improve_prediction(self, info_gains_preds):

        for i in range(5):
            info_gains_preds[i] = np.clip(
                info_gains_preds[i], self.lower_bound, self.upper_bound
            )

        info_gains_preds[5] = np.clip(info_gains_preds[5], 0, 1)

        info_gains_new = copy.deepcopy(info_gains_preds)
        info_gains_preds = copy.deepcopy(info_gains_preds)
        for i in range(self.d.shape[0]):
            for j in range(self.d.shape[1]):
                subgroup_id = self.subgroup_func.get_subgroup(info_gains_preds[i][j])

                info_gains_new[i][j] = (
                    info_gains_preds[i][j] + self.d[i][j][subgroup_id]
                )

        return info_gains_new


class online_gd_changing_lr:
    def __init__(
        self,
        path_length=40,
        num_bins=50,
        lower_bound=0,
        upper_bound=5,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_bins = num_bins
        self.subgroup_func = subgroup_func(lower_bound, upper_bound, num_bins)
        self.d = np.zeros((6, path_length, num_bins)).astype(np.float32)

        self.count = np.ones((6, path_length, num_bins))

    # def get_d(self, current_info_gain_context):
    #     curr_d = []
    #     for i, ig in enumerate(current_info_gain_context):
    #     curr_d.append(self.d[i, int(ig // self.group_interval)])
    #     return curr_d

    # def increase_count(self, current_info_gain_context):
    #     for i, ig in enumerate(current_info_gain_context):
    #     self.count[i, int(ig // self.group_interval)] += 1

    def update_d(self, info_gain_feedbacks, info_gain_preds, info_gain_hat):

        for i in range(5):
            info_gain_feedbacks[i] = np.clip(
                info_gain_feedbacks[i], self.lower_bound, self.upper_bound
            )
            info_gain_preds[i] = np.clip(
                info_gain_preds[i], self.lower_bound, self.upper_bound
            )
            info_gain_hat[i] = np.clip(
                info_gain_hat[i], self.lower_bound, self.upper_bound
            )

        info_gain_feedbacks[5] = np.clip(info_gain_feedbacks[5], 0, 1)
        info_gain_preds[5] = np.clip(info_gain_preds[5], 0, 1)
        info_gain_hat[5] = np.clip(info_gain_hat[5], 0, 1)

        # curr_d = copy.deepcopy(self.d)
        # self.increase_count(current_info_gain_context)

        # actual_d = info_gain_feedbacks - info_gain_hat

        # d_diff = actual_d - curr_d

        for i in range(self.d.shape[0]):
            for j in range(self.d.shape[1]):
                if i == 5:
                    subgroup_id = self.subgroup_func.get_subgroup_w_range(
                        info_gain_preds[i][j], 0, 1
                    )
                    upper_bound = 1
                    lower_bound = 0
                else:
                    subgroup_id = self.subgroup_func.get_subgroup(info_gain_preds[i][j])
                    upper_bound = self.upper_bound
                    lower_bound = self.lower_bound

                for k in range(self.d.shape[2]):
                    if np.sum(subgroup_id == k) == 0:
                        continue

                    # d_k = np.mean(actual_d[i][j][subgroup_id == k])
                    # d_diff = d_k - curr_d[i][j][k]
                    actual_g_k = np.mean(info_gain_feedbacks[i][j][subgroup_id == k])
                    improved_g_k = np.mean(info_gain_hat[i][j][subgroup_id == k])

                    curr_gradient = 1 if actual_g_k > improved_g_k else -1

                    self.count[i, j, k] += np.sum(subgroup_id == k)
                    curr_lr = (
                        np.sqrt(1 / self.count[i, j, k])
                        * (upper_bound - lower_bound)
                        / 2
                    )
                    last_lr = (
                        np.sqrt(1 / (self.count[i, j, k] - 1))
                        * (upper_bound - lower_bound)
                        / 2
                    )
                    self.d[i][j][k] = (
                        self.d[i][j][k] * curr_lr / last_lr
                        + curr_gradient * curr_lr / 2
                    )

    def improve_prediction(self, info_gains_preds):

        for i in range(5):
            info_gains_preds[i] = np.clip(
                info_gains_preds[i], self.lower_bound, self.upper_bound
            )

        info_gains_preds[5] = np.clip(info_gains_preds[5], 0, 1)

        info_gains_new = copy.deepcopy(info_gains_preds)
        info_gains_preds = copy.deepcopy(info_gains_preds)
        for i in range(self.d.shape[0]):
            for j in range(self.d.shape[1]):
                if i == 5:
                    subgroup_id = self.subgroup_func.get_subgroup_w_range(
                        info_gains_preds[i][j], 0, 1
                    )
                else:
                    subgroup_id = self.subgroup_func.get_subgroup(
                        info_gains_preds[i][j]
                    )

                info_gains_new[i][j] = (
                    info_gains_preds[i][j] + self.d[i][j][subgroup_id]
                )

        return info_gains_new
