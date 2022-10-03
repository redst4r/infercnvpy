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

from .priority_queue import Priority
class PWC_Segmentor_Heap():

    def __init__(self, signal: np.ndarray, verbose=False):
        """
        signal: 1D signal to get segmented
        """
        assert len(signal.shape) == 1, "signal needs to be 1D. flatten it?"
        # each interval is represented by its start and end
        # i.e. interval[i] has interval_start[i] and interval_end[i]
        self.heap = PriorityHeap(max_size=len(signal), verbose=verbose)
        self.signal = signal 
        self.verbose = verbose
        self.interval_start= np.arange(0, len(signal))
        self.interval_end = np.arange(1, len(signal)+1)
        self.interval_id = np.array([f"I_{_}" for _ in np.arange(0, len(signal))])  # by convention, if we merge two intervals, we keep the name of the left
        
        for i in self.interval_start[:-1]:  # the last/most-right interval cant be merged, notthing right of it!
            priority = (-1) * self._lambda_hat(i)  # -1 for working with a maxHeap
            self.heap.insert(priority, f'I_{i}')
            
                                 
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
        
        assume a layout like this 1|2|3|4  and a merge of 2 -> 1|2  |4
        
        after the merge, we have to update the cost/priority
        - the interval[left_index] gets new cost (potentially merging it to 4 )
        - the interval to the left (1 in the exmaple) also needs updating
        - interval 3 (being merged to 2) will be removed entirely
        """
        right_ix = left_ix + 1
    
        # now, for the interaction with the HEAP, where we need the interval names
        # important: do this BEFORE deleting anything in self.interval_ids
        right_ID = self.interval_id[right_ix]
        left_ID = self.interval_id[left_ix]        

        
        # set the left-interval's end to the right-intervals end 
        self.interval_end[left_ix] = self.interval_end[right_ix]
        
        
        # delete the right interval from the heap
        # there's one weird exception: Merging into the rightmost interval, i.e. 1|2|3|4
        # merging 3|4
        # the rightmost interval (4) is NOT on the heap (since it cant be merged to the right)
        # its still a valid merge (the interval exists)
        # the new merged interval 3 (1|2|3) is now the new rightmost interval
        # - remove it from the stack, it has no priority
        # TODO this is really UGLY, with the order of deleting the array etc!!!
        if right_ID == self.interval_id[-1]:
            # no need to delete the right itnerval from the heap
            # remove the left, as it will never be merged
            heap_ix = self.heap.search_element(left_ID)
            self.heap.Remove(heap_ix)
            
            # delete the right interval from the array
            # note the palcement of this code. AFTER THIS, the right_ix wont be a valid address any more!
            # the left_ix doesnt change
            self.interval_start = np.delete(self.interval_start, right_ix)
            self.interval_end = np.delete(self.interval_end, right_ix)
            self.interval_id = np.delete(self.interval_id, right_ix)
            del right_ix, right_ID
            
            # no need to update the left priority, its not on the heap anymore
            
        else:
            heap_ix = self.heap.search_element(right_ID)
            self.heap.Remove(heap_ix)
        
        
            # delete the right interval from the array
            # note the palcement of this code. AFTER THIS, the right_ix wont be a valid address any more!
            # the left_ix doesnt change
            self.interval_start = np.delete(self.interval_start, right_ix)
            self.interval_end = np.delete(self.interval_end, right_ix)
            self.interval_id = np.delete(self.interval_id, right_ix)
            del right_ix, right_ID

            # update the left interval on the heap, its priority changed
            newPriority = (-1) * self._lambda_hat(left_ix)  # -1 to make it work with a max heap
            # not that it's boundaries changed too, but the name stays the same!! no need to change the item at all
            heap_ix = self.heap.search_element(left_ID)
            self.heap.changePriority(heap_ix, newPriority)      
        
        # update the interval LEFT of left_ix (it got new priority too)
        # if left_ix ==0 (already the leftmost), skip
        if left_ix >0:
            leftleft_ix = left_ix - 1
            leftleft_id = self.interval_id[leftleft_ix]
            heap_ix = self.heap.search_element(leftleft_id)
            
            newPriority = (-1) * self._lambda_hat(leftleft_ix)  # -1 to make it work with a max heap
            self.heap.changePriority(heap_ix, newPriority)      
            
        
        
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
            priority, the_ID = self.heap.getMax()
            Lmin = (-1) * priority  ## again, maxHeap! high priorty -> low cost
            
            _merge_ix = np.where(self.interval_id == the_ID)[0][0]
            
            # sanity check, making sure Heap and data are in sync
#             ptest = -self._lambda_hat(_merge_ix)
#             np.testing.assert_allclose(priority, ptest, err_msg=f"Priorities dont match! {priority} vs {ptest}, node {the_ID}")
            
            
            # if the cost if smaller than our current threshold, do merge
            if Lmin < current_lambda:
                iname = self.interval_id[_merge_ix]
                if self.verbose:
                    print(f'merging {iname}, ix {_merge_ix}')
                    
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
