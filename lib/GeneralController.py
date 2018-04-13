import torch.nn as nn
from .qrnn import QRNN
from . import FLAGS
from torch.nn import functional as F
from torch.distributions import Categorical
import torch
import numpy as np
from torch.autograd import Variable
from ipdb import set_trace
from .CreateChild import build_model

def to_cuda(var_or_tensor):
    if torch.cuda.is_available():
        return var_or_tensor.cuda()
    else:
        return var_or_tensor

def to_var(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)

def uct_choice(curr_node): return curr_node["children"][curr_node["max_uct"]]


def select(root_node):
    curr_node = root_node

    while curr_node.children is not None:
        curr_node = uct_choice(curr_node)

    return curr_node


def U_func(P, N): return P/(1+N)


def Q_func(W, N): return W/N


def backup(expanded_node, value):
    node = expanded_node
    while node["parent"] is not None:
        node = update_node(node, value)
        node = node["parent"]

    return node


def update_node(node, value):
    node["N"] += 1
    node["W"] += value
    node["Q"] = Q_func(W=node["W"], N=node["N"])
    node["U"] = U_func(P=node["P"], N=node["N"])
    UCT = node["Q"] + node["U"]
    if UCT > node["parent"]["max_uct"]["score"]:
        node["parent"]["max_uct"]["score"] = UCT
        node["parent"]["max_uct"]["idx"] = node["idx"]

    return node


def expand(curr_node, policy, value, next_type, next_inputs):
    curr_node["children"] = []
    curr_node["max_uct"] = {}

    max_U = 0
    max_U_idx = None

    for i, p in enumerate(policy):
        U = U_func(p, 0)

        child = {
            "N": 0,
            "W": 0,
            "Q": 0,
            "U": U,
            "P": p,
            "parent": curr_node,
            "idx": i,
            "next_type": next_type,
            "inputs": next_inputs
        }

        curr_node["children"].append(child)

        if U > max_U:
            max_U = U
            max_U_idx = i

    curr_node["max_uct"] = {
        "score": max_U, "idx": max_U_idx
    }

    return curr_node, value


def choose_real_move(node):
    child_visits_probas = node["child_visits"]/node["child_visits"].sum()

    action = np.random.choice(
        node["child_visits"], p=child_visits_probas)

    return action, child_visits_probas


#soooo... here are my thoughts on ENAS
#The paper is pretty unclear in a lot of places, and the code is dense and complicated
#I mostly have it figured out and I think I have the gist, at least of the most important part
#i.e. the parameters sharing.

#the big idea is, use an autoregressive LSTM to make decisions about subgraph of larger graph
#to use, and have separate parameters for each decision

#using that we can pretty simply change it to our own goals
#we need to think about what 

#in the same way we make a decision 

class GeneralController(nn.Module):
    def __init__(self):
        super(GeneralController, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # I dont need to exactly clone this, I just want the general idea
        # If I can use stuff like QRNN's it might be more efficient anyways
        
        self.rnn = QRNN(FLAGS.LSTM_SIZE, FLAGS.LSTM_SIZE, num_layers=FLAGS.NUM_LSTM_LAYERS,
            dropout=.2)

        self.reset_emb = nn.Sequential(*[
                    nn.Linear(1, FLAGS.LSTM_SIZE),
                    nn.LayerNorm(FLAGS.LSTM_SIZE),
                    nn.ReLU()
                ])

        #g_emb
        #This embedding should let the system know a new architecture has been started
        # self.reset_emb = torch.rand(1, 20)

        start_embs = []
        count_embs = []
        start_logits = []
        count_logits = []
        for _ in range(FLAGS.NUM_BRANCHES):
            #may not want relu, not sure
            #well lets see... ideally we do want the embeddings to be trainable right?
            #that way we can change the embeddings
            inner_start_embs = []
            for _ in range(FLAGS.OUT_FILTERS):
                inner_start_embs.append(nn.Sequential(*[
                        nn.Linear(FLAGS.OUT_FILTERS, FLAGS.LSTM_SIZE),
                        nn.LayerNorm(FLAGS.LSTM_SIZE),
                        nn.ReLU()
                    ]))
            
            inner_start_embs = torch.nn.ModuleList(inner_start_embs)
            start_embs.append(inner_start_embs)

            inner_count_embs = []
            for _ in range(FLAGS.OUT_FILTERS - 1):
                inner_count_embs.append(nn.Sequential(*[
                        nn.Linear(FLAGS.OUT_FILTERS - 1, FLAGS.LSTM_SIZE),
                        nn.LayerNorm(FLAGS.LSTM_SIZE),
                        nn.ReLU()
                    ]))

            inner_count_embs = torch.nn.ModuleList(inner_count_embs)

            count_embs.append(inner_count_embs)

            start_logits.append(nn.Linear(FLAGS.LSTM_SIZE, FLAGS.OUT_FILTERS))
            count_logits.append(nn.Linear(FLAGS.LSTM_SIZE, FLAGS.OUT_FILTERS - 1))

        self.start_embs = torch.nn.ModuleList(start_embs)
        self.count_embs = torch.nn.ModuleList(count_embs)
        self.start_logits = torch.nn.ModuleList(start_logits)
        self.count_logits = torch.nn.ModuleList(count_logits)

        self.attn1 = nn.Linear(FLAGS.LSTM_SIZE, FLAGS.LSTM_SIZE)
        self.attn2 = nn.Linear(FLAGS.LSTM_SIZE, FLAGS.LSTM_SIZE)
        self.v_attn = nn.Linear(FLAGS.LSTM_SIZE, 1)

    def forward(self):
        #first embedding receives 1 as an input
        embeddings_input = to_var(torch.ones(1, 1))

        inputs = self.reset_emb(embeddings_input).unsqueeze(0)

        anchors = []
        anchors_w_1 = []

        arc_seq = []
        # entropys = []
        # log_probs = []
        # skip_count = []
        # skip_penalties = []

        # skip_targets = to_cuda(torch.FloatTensor(1 - FLAGS.SKIP_TARGET, FLAGS.SKIP_TARGET))

        for layer_id in range(FLAGS.NUM_LAYERS):
            for branch_id in range(FLAGS.NUM_BRANCHES):
                #start
                x = self.rnn(inputs)[0].view(1, -1)
                start_logits = self.start_logits[branch_id](x)
                start_probas = F.softmax(start_logits.squeeze(), 0)
                start_probas_np = start_probas.detach().data.numpy()
                start_idx = np.random.choice(FLAGS.OUT_FILTERS, p = start_probas_np)
                arc_seq.append(start_idx)
                #soooo in their one they basically have a different slice
                #or a different linear layer for each of the inputs
                #so if we have a shape (num_branches, out_filters, lstm_size)
                #aka (4, 10, 32)
                #we want [0, start_idx] to select which linear
                #and then we are left with 32 dims
                #in theirs they only have num_branches number of linears
                #and then they take slices of it?
                #so basically for each branch there is a different linear
                #and then for each linear we can control what we input
                #so idk, the main idea is we have a different linear for each branch
                #and start combo. 
                #to achieve that we could just do a nested list
                #or we could just get the input with the start logits, and 
                #then slice based on the start_idx (better I think)

                #well whatever let me do what make sense to me
                #basically we want a linear layer which will acts as an embedding
                #for a certain start choice
                #I think that for now we can ignore the start_idx indexing
                inputs = self.start_embs[branch_id][start_idx](start_logits).unsqueeze(0)

                #[start_idx]

                #count
                x = self.rnn(inputs)[0].view(1, -1)
                count_logits = self.count_logits[branch_id](x)
                mask = torch.linspace(0, FLAGS.OUT_FILTERS-2, FLAGS.OUT_FILTERS-1)
                zeros = torch.zeros(FLAGS.OUT_FILTERS-1)

                #still dont totally understand this where, just copied it
                #look it over more closely later
                count_probas = F.softmax(count_logits.squeeze(), dim=0)
                count_probas = torch.where(mask <= FLAGS.OUT_FILTERS - 1 - start_idx, 
                    count_probas, zeros)
                count_probas /= count_probas.sum()

                count_probas_np = count_probas.detach().data.numpy()
                count_idx = np.random.choice(FLAGS.OUT_FILTERS-1, p = count_probas_np)
                arc_seq.append(count_idx + 1)
                #[count_idx]
                #so basically I input the logits, then get the slice based on what
                #I want

                #going to leave it the same for now, seems like memory intensive or
                #redundant calculation to do it the other way
                inputs = self.count_embs[branch_id][count_idx](count_logits).unsqueeze(0)

            x = self.rnn(inputs)[0].view(1, -1)

            if layer_id > 0:
                query = torch.cat(anchors_w_1, dim=0)
                query = self.tanh(query + self.attn2(x))
                query = self.v_attn(query)

                skip_logits = torch.cat([-query, query], dim=1)
                skip_probas = F.softmax(skip_logits, dim=1)
                skip_probas_np = skip_probas.detach().data.numpy()
                skip_seq = []
                for skip_proba in skip_probas_np:
                    skip_idx = np.random.choice(2, p=skip_proba)
                    arc_seq.append(skip_idx)
                    skip_seq.append(skip_idx)

                skips = torch.from_numpy(np.array(skip_seq)).float().unsqueeze(0)
                inputs = skips.mm(torch.cat(anchors))

                inputs /= (1.0 + skips.sum())
                inputs = inputs.unsqueeze(0)
            else:
                inputs = self.reset_emb(embeddings_input).unsqueeze(0)
            
            anchors.append(x)
            anchors_w_1.append(self.attn1(x))
        

        #I dont really know how this work yet
        #for now I think we need to separate the skips 
        #although actually I guess it is separated already because every layer has a 
        #certain number of skip options (0 = 0, 1 =1, etc)
        sample_arc = torch.from_numpy(np.array(arc_seq).astype("float32"))

        model = build_model(sample_arc)
        return model, sample_arc
        # child(input)

        #call net creator with sample arc




        # so lets see, ideally we want to keep the tree over time
        # so every time we run forward we make a new tree
        # and then after each decision point we throw away the top

    # soooo theres no doubt about this that it is really complicated
    # I need to basically just try it out like I did with MCTS and alphazero until I got it
    # just keep going through it, put what makes sense, and get the main idea
    # the main idea is sharing parameters.

    def start_expansion(self, inputs, branch_id):
        start_qrnn = self.qrnn(inputs)
        start_logits = self.start[branch_id](start_qrnn)
        start_log_probas = F.log_softmax(start_logits, axis=1)

        start_critic_input = start_qrnn.view(
            inputs[0].size()[0], -1)

        probas = F.softmax(start_logits, axis=1)
        value = self.start_critic(start_critic_input)

        return probas, value, start_log_probas

    def count_expansion(self, inputs, branch_id):
        count_qrnn = self.qrnn(inputs)
        count_logits = self.count[branch_id](count_qrnn)
        count_log_probas = F.log_softmax(count_logits, axis=1)

        count_critic_input = count_qrnn.view(inputs[0].size()[0], -1)

        probas = F.softmax(count_logits, axis=1)
        value = self.count_critic(count_critic_input)

        return probas, value, count_log_probas

    # sooo it is autoregressive where each decision is fed into the next input
    #hmm in theory I could copy the beginning to before the first decision is made
    #and then have each simulation reset back to that point
    #that way it would be easy to restore from the beginning, and have it be if the simulations
    #didn't happen, and each simulation would be as if it had started from there
    #so basically I would copy the nets parameters right before the first decision
    #could it be earlier on for convenience?
    #it would be great if i could just do it on the outside, i.e. do it around the whole function
    #so basically the value function is produced at the end, in theory from the memory 
    #of what actions it just selected, and then that is backpropagated?
    #we want the value to be at every decision if we want to use an alpha zero framework though
    #so at each decision we produce a value and a probability, and we pass those with an 
    #input to the alpha zero function. to accurately get the value function we need to 
    #run the net as if it were doing it normally.
    #in this system all we need to get through is the sampled index, so we just need
    #the uct selected index which will move use to the next decision
    #each type of decision

    #so from a high level view what we want is to simulate going through the forward function
    #multiple times, and use the values and visits to those future states to dictate our final
    #move selection, so maybe we can even wrap the whole forward function
    #we pass it a root node, go through the whole thing, and update the tree
    #based on any results we get along the way. 
    #we could have multiple values at each decision point, or....
    #well lets see, I think we do need a value and a probability at each leaf node
    #a leaf node is going to 

    #basically we go through the function, we get to a decision, and then we 
    #pass is to the alpha zero function, which takes....

    #side note about the valid idea:
    #Basically the idea is use MCTSnet, but change it from using 
    #separate nets to all in one
    #so for example we use a QRNN 
    #it receives a state, and outputs an imagined state, and whether the state is valid or not,
    # valid is a subset of real, since some real game boards could be invalid
    # so basically the QRNN learns if it is a valid state
    # to prevent the QRNN from remembering that it produced the state we could 
    # copy network or use the best network to produce the next state

    #well whatever, imagine we trained the net to produce valid states
    #the goal is to have it imagine a future valid state, remember something about it
    #and use it to choose a best possible policy.
    #an aux policy loss would be comparing the policy for the first net after some number of 
    #sims to when it had no sims, i.e. training the raw network probas to match the
    #search probas. The idea of it is to enforce the net to image valid states,
    #which would be in the same format in the original state and allow it to be reused
    #with the same net. in theory it should be able to imagine any future state, think about
    #the value/policy/whatever and use that to choose the original policy

    #in the case where there isn't a "valid state" i.e. it doesnt have enforced rules
    #we could probably use a is_real normal gan loss, to at enforoce that the representations
    #learned are in the same domain as the real state inputs, that will at least allow 
    #to use the same net to process the inputs the same way

    #so for example with this ENAS, we could have a net, maybe a QRNN that
    #takes a state and outputs whether it is real or not, a policy for the original state,
    #a value for the original state (optional) and an imagined state to inspect next
    #another possible head is whether or not to do another simulation and find some type of
    #objective which will trade off value for num_sims
    #maybe something like that, the loss = ((reward + value)/2)*(num_sims**2)
    #hmmmm.... well that's not exactly what I want. The value of the state could be low 
    #and the number of the sims..
    #ideally you would want to be able to think about anything to improve your internal 
    #memory (the QRNN memory matrix), 
    #maybe just a sigmoid about whether or not to do another sim, and then you use the value of 
    #the next state or the reward
    #in theory you could use the reward - the value of the next state or something
    #so the more surprised you are about the value the more 
    #so the loss would be the output of the percentage, e.g. .7
    #if the reward-value
    #so if (1 - .7)*(num_sims**2)
    #the reward was better than expected, so it is positive meaning that the loss
    #would be 
    #maybe I could use an alphazero loss to the numbr of sims lol
    #just getting a bit kookie, at some point I need to have a loss that isn't like ethat
    #maybe stick it on one of the other losses, so like the
    # maybe -log (1-sim_chance)
    # so if sim_chance is close to 0 we will get a small number
    # adn if sim_chance is close to 1 we will get a big number
    # #and then we need to scale it by the reward
    # reward-value
    # if the value was better than expected 
    #anyways, it could just be empirically chosen for now
    #maybe at some point this could all be meta optimized

    #I wish I could think of a way to simplify it though
    #if I can wrap it in the forward it would be way easier
    #can we just run through a forward, accumulate values, and 
    #re run it any desired number of times?
    #so basically we have a Simulation=True which activates alpha zero
    #at each decision we make a UCT choice from the possible actions
    #so basically we would have three root nodes
    #skip_root, start_root, count_root
    #if we hit start for example, we do a UCT choice if it results in a leaf node
    #we expand it and update all nodes in the trajectory
    #okay so where were we so we do a UCT 

    #so we go through forward, we hit a decision point
    #the LSTM has a value function that we reuse base on the most recent input
    #i.e. it kind of uses it's memory

    #so we hit start, we expand it
    #we reset the forward?
    #a bit inefficient, because we are redoing a bunch of stuff 
    #that we dont need to

    #so lets imagine we dont
    #we hit start, we do a bunch of sims...
    #now again the issue is we have interleaving decisions,
    #so to really do a full trajectory they need to be interwoven

    #because of course we need to see what the result will be
    #well, one possible option is that we could do a UCT choice for each of the moves
    #or expand and then do a UCT choice if it hasnt been visited yet

    #so lets see, we go through the forward function, we hit a decision
    #if it hasnt been visited yet we expand it and backpropagate the result to the root,
    #but we dont change selection from the leaf node, instead we do another UCT select with it
    #and continue on

    #then we continue onwards to the next decision, we are now are a branch which will be of
    #type count, so it will produce a different sized probas 

    #if it has been visited before we UCT select, otherwise we backpropagate then UCT select
    #we do that for each decision point and run through the whole forward pass for NUM_SIMS
    #times, with simulation=True specified, which means that we maybe disable some redundant stuff
    # and also when simulation = False we do the selection using the tree, i.e. using the MCTS search probas
    # and we compare those probas to the normal output of the network

    #now, the question is can we do that in the original project?
    #it's feasible I think but it's hard, lets try for a bit

    #in theory they dont reset their net everytime they call build trainer I dont think
    # def forward(self, orig_inputs):
    #     # so every decision should have a vlaue and a probability
    #     arc_seq = []
    #     anchors = []
    #     anchors_w_1 = []

    #     inputs = self.g_emb(orig_inputs)

    #     # at the start of the MCTS sims I may want to save the parameters of the
    #     # net, and then at the start of each trajectory I restart them
    #     root_node = {"parent": None, "children": None,
    #                  "next_type": "start", "inputs": inputs}

    #     #Just to think about it, how would this work it if was batched.
    #     #in theory each batch_idx would be a probability and a value, and they all could
    #     #diverge, and in effect we would be making a batch worth of separate models
    #     #I think assuming batch_size = 1 for now is fine
    #     for layer_id in range(FLAGS.NUM_LAYERS):
    #         for branch_id in range(FLAGS.NUM_BRANCHES):
    #             for sim_id in range(FLAGS.NUM_SIMS):
    #                 leaf_node = select(root_node)

    #                 if leaf_node["next_type"] is "start":
    #                     probas, value, log_probas = \
    #                         self.start_expansion(
    #                             leaf_node["inputs"], branch_id)
    #                     next_type = "count"
    #                     next_inputs = self.start[branch_id].parameters()[
    #                         leaf_node["idx"]]
    #                 else:
    #                     probas, value, log_probas = \
    #                         self.count_expansion(
    #                             leaf_node["inputs"], branch_id)
    #                     next_inputs = self.count[branch_id].parameters()[
    #                         leaf_node["idx"]]
    #                     next_type = "start"

    #                 if sim_id == 0:
    #                     first_log_probas = log_probas

    #                 expanded_node = expand(
    #                     leaf_node, probas, value, next_type, next_inputs)

    #                 root_node = backup(expanded_node, value)

    #             action, search_probas = choose_real_move(root_node)

    #             policy_loss = 0
    #             for search_proba, log_proba in zip(search_probas, first_log_probas):
    #                 search_proba = search_proba.unsqueeze(0)
    #                 log_proba = log_proba.unsqueeze(-1)

    #                 policy_loss += search_proba.mm(log_proba)

    #             policy_loss /= len(first_log_probas)

    #             if root_node["next_type"] is "start":
    #                 inputs = self.count[branch_id].parameters()[action]
    #                 arc_seq.append(action)
    #             else:
    #                 inputs = self.start[branch_id].parameters()[action]
    #                 #not sure why this is action+1, need to read more
    #                 arc_seq.append(action+1)

    #         # Branches done
    #         after_branches_qrnn = self.qrnn(inputs)

    #         #so I can do the bucketed alpha zero search here, or I can skip that for now
    #         #how hard would it be
    #         #I just need to take the one probability for skip chance, and bucket into whatever
    #         #interval I care about, for example 10 different buckets
    #         #then treat those as probability choices for UCT
    #         #One question is how do we get backwarsd

    #         if layer_id > 0:
    #             root_node = {"parent": None, "children": None,
    #                  "next_type": "start", "inputs": inputs}

    #             query = torch.stack(anchors_w_1)
    #             query = self.tanh(query + self.w_attn_2(after_branches_qrnn))

    #             query = self.v_attn(query)
    #             # size one
    #             skip_logit = torch.stack([-query, query])

    #             skip_proba = self.sigmoid(skip_logit)

    #             skip = Categorical(torch.log(skip_proba))
    #             arc_seq.append(skip)

    #             skip = skip.unsqueeze(0)
    #             inputs = skip.mm(torch.stack(anchors))
    #         else:
    #             # could be inputs or something
    #             inputs = self.g_emb(after_branches_qrnn)

    #         anchors.append(after_branches_qrnn)
    #         anchors_w_1.append(self.w_attn_1(after_branches_qrnn))

    #     # Layers Done
    #     arc_seq = torch.stack(arc_seq)

    #     return arc_seq

    #     # I dont really under stand it totally
    #     # it seems like it is getting a probability distribution
    #     # this is just a low level iterating through each of the placeholder variables
    #     # seems like she couldve maybe reused it, idk
    #     # but anyways what I effectively want is for it to produce one for each branch
