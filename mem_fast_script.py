
# depend on precision.
OPTIMIZER_SCALE=12

def mem_str(mem):
    mem = mem/1024.0/1024.0
    if (mem<1024.0):
        return f"{mem} MB"
    else:
        return f"{mem/1024.0} GB"

def fixed_params(hidden_size,layers,vocab_size=0):
    params = layers*(12*hidden_size**2+13*hidden_size)+ vocab_size*hidden_size
    return params

def rank_basic_mem(hidden_size,num_heads,layers,seq_len=2048,batch=1,tp_size=1,pp_size=1,dp_size=1,zero_stage=0,vocab_size=0,):
    # TODO: multi-group-queue
    params = fixed_params(hidden_size/tp_size,layers/pp_size,vocab_size)
    # num_ranks=tp_size*pp_size*dp_size
    zero_map=[OPTIMIZER_SCALE+4,4+12/dp_size,2+(2+OPTIMIZER_SCALE)/dp_size,(4+OPTIMIZER_SCALE)/dp_size]

    zero_map=[16,4+12/dp_size,2+12/dp_size,16/dp_size]
    rank_params_mem = zero_map[zero_stage]*params
    
    
    rank_activation_mem=train_runtime_activation_mem(hidden_size/tp_size,num_heads/tp_size,layers/pp_size,seq_len,batch,vocab_size)
    print(f"rank_params:{params/1000000000.0}B")
    print(f"rank_fixed_mem:{mem_str(rank_params_mem)}")    
    print(f"rank_train_activation_mem:{mem_str(rank_activation_mem)}")
    # print_mem(rank_params_mem)
    

def train_runtime_activation_mem( hidden_size, num_heads, layers,seq_len=2048,batch=1,vocab_size=0):  

    activation_mem = ((34 * batch * seq_len * hidden_size + 5 * batch * seq_len**2 * num_heads) * layers )+batch*vocab_size*hidden_size
    return activation_mem

# rank_basic_mem(2560,32,32,seq_len=2048,batch=1,tp_size=1,pp_size=8,dp_size=1,zero_stage=1,vocab_size=50000)


print(mem_str(train_runtime_activation_mem(4096,32,28,2048)))
