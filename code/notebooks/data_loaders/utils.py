from re import M
import torch
import numpy as np

class Utilities():
    
    # Function to normalize pixels
    @staticmethod 
    def norm_pixel(x):
        x = x.astype('float32')
        return (x*2)/255-1

    # Function to denormalize pixels
    @staticmethod 
    def denorm_pixel(x):
        x = x.astype('float32')
        return (x+1)*255/2

    # Function to normalize pitch and roll angles
    @staticmethod
    def norm_pr(x, min = -90.0, max = 90.0):
        return ((x - min) * 2) / (max - min) - 1

    # Function to denormalize pitch and roll angles
    @staticmethod
    def denorm_pr(x, min = -90.0, max = 90.0):
        return ((x + 1) * (max - min))/2 + min


    # function to measure inference time on GPU
    @staticmethod
    def inference_time(model, dummy_input, repetitions=10000):
        device = torch.device("cuda")
        model.to(device)
        dummy_input.to(device)
        # INIT LOGGERS
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = np.zeros((repetitions, 1))
        #GPU-WARM-UP
        for _ in range(1000):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)

        return timings, mean_syn

    # function to test if all (input, output) pairs are from an equal episode
    @staticmethod
    def test_sequence_integrity(all_sequences):
        i = 0
        for sequence in all_sequences:
            input = sequence[0]
            output = sequence[1]

            episode = input.iloc[0]["episode"]
            episode1 = output.iloc[0]["episode"]
            
            if(episode1 != episode):
                print("Input and output from different episodes")
                break

            if (len(input['episode'].unique()) != 1):
                print("input seq: "+i+" contains data from different episodes")
                break
            
            if (len(output['episode'].unique()) != 1):
                print("output seq: "+i+" contains data from different episodes")
                break
            
            i += 1

    
