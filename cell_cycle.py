import networkx as nx
import numpy as np

def gnetwork():
    genes = ['Cln3','MBF','SBF','Cln1,2','Cdh1','Swi5','Cdc20,14', 'Clb5,6','Sic1','Clb1,2','Mcm1']

    s = [1, 1, 2, 3,  4,  4,   5,  6,  7, 7,  7,  7,  7,  8, 8,  8,   8,  9,  9,  10, 10, 10, 10, 10, 10, 10, 11, 11, 11];
    t = [2, 3, 8, 4,  5,  9,  10, 9,  5,  6,  8,  9, 10, 5, 9, 10, 11, 8, 10,  2,   3,    5,   6,   7,   9,  11,  6 ,  7,  10];
    weights = [1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1 ,-1, 1, 1, 1, 1];

    G = nx.DiGraph()

    G.add_nodes_from(range(1, 12))

    arr = [(s[i], t[i], weights[i]) for i in range(11)]

    G.add_weighted_edges_from(arr)
    
    GRN = nx.adjacency_matrix(G).todense()

    N = GRN.shape[0]

    influence = np.zeros([N, N])

    
    for i in range(len(weights)):
        influence[s[i]-1, t[i]-1] = weights[i]

    return GRN, influence, genes 

#def allcombs(experimentOutcomes):
#
#    experimentOutcomes = np.flip(experimentOutcomes);
#    n = np.size(experimentOutcomes);
#    c = cell(1, n);
#
#    return experimentOutcomes

def allcombs(n, arr = [[0], [1]], i = 1):
	if i == n:
		return np.array(arr)
	else:
		L = len(arr)
		arr_ = []
		for j in range(L):
			arr_.append(arr[j] + [0])
			arr_.append(arr[j] + [1])
		return allcombs(n, arr_, i + 1)

def initialization(network, states):

    N = len(states)

    idxs = np.array(range(len(states)))

    msg = np.zeros((N, N))

    for j in range(N):
        x = np.array(network[j, :]).flatten()
        idx_init = idxs[x == 1]

        for k in range(len(idx_init)):
            msg[j, idx_init[k]] = states[j]

    return msg

def xor(a, b):
    xor = False
    xor= (a or b) and not (a and b)
    return xor

def logic_function(influence, msg_pr, msg_ch):

    if influence == 1:
        upd_msg_ch = float(msg_pr or msg_ch)
    elif influence == -1:
        upd_msg_ch = float(xor(msg_pr, msg_ch) and msg_ch)

    if msg_pr == 0:    # inactive parent genes contribute no information
        upd_msg_ch = []

    return upd_msg_ch

def degradation(previous_state, msg_input):

    x = np.sort(msg_input)
    y = np.unique(x)
    length = []

    for i in range(len(y)):
        length = np.concatenate((length, np.count_nonzero(x == y[i])))

    a = max(length)
    idxs = np.array(range(len(length)))

    idx_max = idxs[length == a]

    if len(idx_max) > 1:
        if previous_state == 1:
            state = 0
        else:
            state = previous_state

    else:
        state = y[idx_max]

    return state

def message_pass_calc(input_msg, previous_state):
    p = len(input_msg)
    states = [0, 1]
    q_len = []

    for i in range(len(states)):
        idx = []
        for j in range(p):
            idx = np.append(idx, np.count_nonzero(input_msg[j] == states[i]))

        q_len = np.append(q_len, len(idx))

    if q_len[0] == q_len[1]:
        msg_out = previous_state

    elif q_len[0] > q_len[1]:
        msg_out = 0
    else:
        msg_out = 1

    return msg_out

def f_node_update(daG_mat, influence, input_msg, nodes):

    N = daG_mat.shape[0]
    fact_msg = np.zeros((N, N))
    idxs = np.array(range(daG_mat.shape[0]))

    for i in range(N):
        x = np.array(daG_mat[:, i]).flatten()
        idx = idxs[x == 1]

        if len(idx) == 1:
            if np.any(nodes == idx) == True and input_msg[idx, i] == 1:
                msg_snd = 0
            else:
                msg_snd = input_msg[idx, i]

            fact_msg[idx, i] = msg_snd
        else:
            for j in range(len(idx)):
                if idx[j] == i:
                    tmp_msg = []
                    new_idx = idx[idx != i]

                    for p in range(len(new_idx)):
                        upd_msg = logic_function(influence[new_idx[p],i], input_msg[new_idx[p],i], input_msg[i,i])
                        tmp_msg = [tmp_msg, upd_msg]

                    if np.any(nodes == i) == True:
                        if (len(tmp_msg) == 0 and input_msg[i, i] == 1) or (len(tmp_msg) == 0 and input_msg[i, i] == 0):
                            msg_snd = 0
                        else:
                            msg_snd = degradation(input_msg[i, i], tmp_msg)


                    else:
                        if len(tmp_msg) == 0:
                            msg_snd = input_msg[i, i]
                        else:
                            msg_snd = message_pass_calc(tmp_msg, input_msg[i, i])


                    fact_msg[i, i] = msg_snd



    return fact_msg

def v_node_update(daG_mat, input_msg):
    N = daG_mat.shape[0]
    states = []

    for i in range(N):
        idx = np.where(daG_mat[i,:]==1)
        idx = idx[1]
        msg_snd = [input_msg[i, i]]
        states = np.concatenate((states, msg_snd))

    var_msg = initialization(daG_mat, states)

    return [var_msg, states]

def unique_counts(msg):
    a,b,c = np.unique(msg,return_index=True,return_inverse=True)
    idx = []
    for i in range(len(b)):
        idx = np.concatenate((idx, np.count_nonzero(c == i)))
    
    val = max(idx)
    idxs = np.array(range(len(idx)))
    
    idx_max = idxs[idx == val]
    if range(len(idx_max)) > 1:
        a_sample = np.random.rand(idx_max,1)
        out_states = a[a_sample, :]
    else:
        out_states = a[idx_max, :]
    
    return out_states

filename = 'cellcycle_data.csv'
data = np.loadtxt(open(filename, "rb"), delimiter=",")

GRN, influence, genes = gnetwork()

full_GRN = GRN + np.identity(GRN.shape[0])

id_nodes = [1, 4, 6, 7, 11];

k = 2;

N = full_GRN.shape[0]

states = allcombs(N)

numStates = states.shape[0]

iterations = 1
fixd_points = []

for i in range(numStates):
    
    init_state = states[i, :]
    
    tmp_fixd_pts = []

    # Realizamos un numero de iteraciones por estado de proteína de red
    for p in range(iterations):
        original_states = init_state
        var_msg = initialization(full_GRN, init_state)
        count = 0

        test = False

        while test == False & count < 50:
            
            tmp_msg = var_msg
            previous_msg = original_states

            fact_msg = f_node_update(full_GRN, influence, tmp_msg, id_nodes)
            [var_msg, updt_states] = v_node_update(full_GRN, fact_msg)

            test = (previous_msg == updt_states).all()

            original_states = updt_states
            count += 1
        
        tmp_fixd_pts = [tmp_fixd_pts, updt_states]

    fixd_point = np.vstack((fixd_points, unique_counts(tmp_fixd_pts)))

[global_attractors, C, D] = np.unique(fixd_points, return_index=True,return_inverse=True)
basis_cnts = []
attractor_membership = []
for i in range(len(C)):
    idxs = np.array(range(len(D)))
    tmp = idxs[D == i]
    basis_cnts = [basis_cnts, len(tmp)]
    attractor_membership[i] = states[tmp, :]
    
global_attractors
basis_cnts