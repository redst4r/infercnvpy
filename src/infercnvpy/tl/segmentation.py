import numpy as np
import matplotlib.pyplot as plt


class PWC_Segmentor():
    
    def __init__(self, signal: np.ndarray):
        """
        signal: 1D signal to get segmented
        """
        assert len(signal.shape) == 1, "signal needs to be 1D. flatten it?"
        # each interval is represented by its start and end
        # i.e. interval[i] has interval_start[i] and interval_end[i]
        self.interval_start= np.arange(0, len(signal))
        self.interval_end = np.arange(1, len(signal)+1)
        self.signal = signal 

    def _get_interval(self, i: int):
        """
        get the signal in the given interval
        """
        return self.signal[self.interval_start[i]:self.interval_end[i]]

    def _get_interval_length(self, i):
        """
        get the length of the interval[i]
        """
        L = self.interval_end[i] - self.interval_start[i]
        assert L == len(self._get_interval(i))
        return L

    def _merge_interval(self, left_ix):
        """
        merge two adajacent intervals
        Convention: when merging, we always merge the left to the right
        i.e. we index the breakpoint by the end of the left interval
        """

        right_ix = left_ix + 1
        # set the left-interval's end to the right-intervals end 
        self.interval_end[left_ix] = self.interval_end[right_ix]
        # delete the right interval
        self.interval_start = np.delete(self.interval_start, right_ix)
        self.interval_end = np.delete(self.interval_end, right_ix)
        
    def _lambda_hat(self, left_ix):
        """
        calculate the cost of merging the left interval (to its right)
        """
        # breakpoint is the end of the left interval
        right_ix = left_ix + 1
        
        R_im1 = self._get_interval_length(left_ix)
        R_i = self._get_interval_length(right_ix)
        U_im1 = self._get_interval(left_ix).mean()
        U_i = self._get_interval(right_ix).mean()
        
        l = ((U_im1 - U_i)**2)*(R_im1 * R_i) / (R_im1 + R_i)
        
        return l

    def plot_current(self):
        """
        plot the segmentation over the signal
        """
        seg_signal = self.get_segmented_signal()
        plt.plot(np.arange(len(self.signal)), seg_signal)
        plt.plot(np.arange(len(self.signal)), self.signal)


    def do_segmentation(self, iterations=10000, beta=0.5):
        """
        run the actual segmentation for a given number of iterations or until we hit the stop criterion
        :return: 
            the schedule of lambda (how its increasing with iterations). 
            When it starts increasing too much, we cant find good merges, and are probably done
        """
        # standard dev of the data, used for the stoppoing criterion
        v = np.std(self.signal)

        # remember how lambda increases
        lambda_schedule = [] 
        current_lambda = 0
        for i in range(iterations):
            
            # find the breakpoint with the smallest lambda
            lambda_vec = np.array([self._lambda_hat(_) for _ in range(0,len(self.interval_start)-1)])
            _merge_ix = np.argmin(lambda_vec)
            
            Lmin = lambda_vec[_merge_ix]  # the cost of mergin
            # if the cost if smaller than our current threshold, do merge
            if Lmin < current_lambda:
                self._merge_interval(_merge_ix)
            else:
                lambda_schedule.append(current_lambda)
                current_lambda = Lmin + 1e-6
            
                # check if we converged
                deltaL = current_lambda - lambda_schedule[-1]
                if deltaL > beta * v: #NOTE small delta values are GOOD, meaning that we're still joining meaninful regions
                    break
        return lambda_schedule

    def get_segmented_signal(self):

        mean_values = []
        for i in range(len(self.interval_start)):
            m = self._get_interval(i).mean()
            mean_values.extend([m]*self._get_interval_length(i))
        mean_values = np.array(mean_values)
        assert len(mean_values) == len(self.signal)
        return mean_values
