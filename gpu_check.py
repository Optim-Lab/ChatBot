#%%
import fire
import GPUtil

def check_gpus_in_use(required_gpus):
    gpus = GPUtil.getGPUs()
    
    inuse = gpus[required_gpus].load

    print(f"Is GPU {required_gpus} available?")
    
    if inuse > 0:
        print("> NO...")
    else:
        print("> YES!!!")
#%%
if __name__ == '__main__':
    fire.Fire(check_gpus_in_use)
#%%