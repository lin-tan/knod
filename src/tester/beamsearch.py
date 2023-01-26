import torch


class BeamSearch:
    def __init__(self, model, node_vocabulary, edge_vocabulary, beam_size):
        self.node_vocabulary = node_vocabulary
        self.edge_vocabulary = edge_vocabulary
        self.beam_size = beam_size
        self.father_beam_size = 1
        self.edge_beam_size = 1
        self.max_step = 128
        self.model = model

    def generate(self, inputs, max_step=None, father_beam_size=3, edge_beam_size=3, identifier_semantics=None):
        hypothesis = []
        encoder_out = self.model.encode(inputs)

        if max_step:
            self.max_step = min(128, max_step)

        inputs = {k: v.to(self.model.gpu2) for k, v in inputs.items()}
        nodes = inputs['nodes'].repeat(self.beam_size, 1)
        target_nodes = torch.zeros(self.beam_size, self.max_step).type_as(inputs['target_nodes'])
        target_nodes[:, 0] = inputs['target_nodes'][0, 0]
        target_out_fathers = torch.zeros(self.beam_size, self.max_step).type_as(inputs['target_out_fathers'])
        target_out_edges = torch.zeros(self.beam_size, self.max_step).type_as(inputs['target_out_edges'])
        target_edges = torch.zeros(self.beam_size, self.max_step, self.max_step).type_as(inputs['target_edges'])
        target_edges[:, 0, 0] = self.edge_vocabulary.self_loop_index
        final_scores = torch.zeros(self.beam_size).type_as(encoder_out)

        father2children = [{} for _ in range(self.beam_size)]

        self.edge_beam_size = edge_beam_size
        src_len = len(inputs['nodes'].masked_select(mask=inputs['rem_tags'] > 1))
        for step in range(self.max_step - 1):
            print('step:', step)
            self.father_beam_size = min(step + 1, father_beam_size)
            if step == 0:
                decoder_input = {
                    'nodes': nodes[:1, :],
                    'target_nodes': target_nodes[:1, :step + 1],
                    'next_father_indices': None,
                    'next_edges': None,
                    'target_edges': target_edges[:1, :step + 1, :step + 1],
                    'rem_tags': inputs['rem_tags']
                }
                beam_ids, next_father_indices, next_edges, final_scores, out_nodes = self.model.decode(
                    encoder_out,
                    decoder_input,
                    edge_of_rem=inputs['target_out_edges'][0, 0],
                    father_beam_size=self.father_beam_size,
                    edge_beam_size=self.edge_beam_size,
                    beam_size=self.beam_size,
                    identifier_semantics=identifier_semantics,
                    # src_len=src_len
                )
                node_scores, node_indices = out_nodes.topk(k=self.beam_size, dim=-1)
                node_scores, node_indices = node_scores.view(-1), node_indices.view(-1)
                final_scores = final_scores.repeat_interleave(self.beam_size, dim=0) + node_scores
                cand_final_scores, sort_order = final_scores.sort(descending=True)
                cand_final_scores = cand_final_scores[: self.beam_size]
                sort_order = sort_order[: self.beam_size]

                next_father_indices = next_father_indices.repeat_interleave(self.beam_size, dim=0)[sort_order]
                next_edges = next_edges.repeat_interleave(self.beam_size, dim=0)[sort_order]

                father_indices = next_father_indices
                edge_indices = next_edges
                node_indices = node_indices.index_select(0, sort_order)

                target_nodes[:, step + 1] = node_indices
                for i in range(target_edges.size(0)):
                    target_edges[i, step + 1, father_indices[i]] = \
                        target_edges[i, father_indices[i], step + 1] = edge_indices[i]
                    target_edges[i, step + 1, step + 1] = self.edge_vocabulary.self_loop_index  # self loop
                target_out_fathers[:, step] = father_indices
                target_out_edges[:, step] = edge_indices

                for i in range(target_out_fathers.size(0)):
                    father2children[i][int(father_indices[i])] = [step + 1]

                final_scores = cand_final_scores
                continue

            decoder_input = {
                'nodes': nodes,
                'target_nodes': target_nodes[:, :step + 1],
                'next_father_indices': target_out_fathers[:, : step],
                'next_edges': target_out_edges[:, : step],
                'target_edges': target_edges[:, :step + 1, :step + 1],
                'rem_tags': inputs['rem_tags']
            }
            beam_ids, next_father_indices, next_edges, final_scores, out_nodes = self.model.decode(
                encoder_out,
                decoder_input,
                scores=final_scores,
                father_beam_size=self.father_beam_size,
                edge_beam_size=self.edge_beam_size,
                beam_size=self.beam_size,
                identifier_semantics=identifier_semantics,
                # src_len=src_len
            )

            node_scores, node_indices = out_nodes.topk(k=self.beam_size, dim=-1)
            node_scores, node_indices = node_scores.view(-1), node_indices.view(-1)
            final_scores = final_scores.repeat_interleave(self.beam_size, dim=0) + node_scores
            cand_final_scores, sort_order = final_scores.sort(descending=True)
            beam_ids = beam_ids.repeat_interleave(self.beam_size, dim=0)[sort_order]

            next_father_indices = next_father_indices.repeat_interleave(self.beam_size, dim=0)[sort_order]
            next_edges = next_edges.repeat_interleave(self.beam_size, dim=0)[sort_order]
            node_indices = node_indices.index_select(0, sort_order)

            eos_mask = node_indices[: self.beam_size].eq(self.node_vocabulary.eos_index)
            eos_cand_final_scores = cand_final_scores[: self.beam_size].masked_select(mask=eos_mask)
            if eos_cand_final_scores.size(0) > 0:
                eos_beam_ids = beam_ids[: self.beam_size].masked_select(mask=eos_mask)
                eos_beam_nodes = target_nodes[eos_beam_ids, :]
                eos_beam_out_fathers = target_out_fathers[eos_beam_ids, :]
                eos_beam_out_edges = target_out_edges[eos_beam_ids, :]
                eos_beam_nodes[:, step + 1] = self.node_vocabulary.eos_index
                eos_beam_out_fathers[:, step] = next_father_indices[: self.beam_size].masked_select(mask=eos_mask)
                eos_beam_out_edges[:, step] = next_edges[: self.beam_size].masked_select(mask=eos_mask)

                for i in range(eos_beam_ids.size(0)):
                    hypothesis.append(
                        {
                            'fathers': eos_beam_out_fathers[i, : step + 1],
                            'edges': eos_beam_out_edges[i, : step + 1],
                            'nodes': eos_beam_nodes[i, 1:step + 2],
                            'final_score': round(float(eos_cand_final_scores[i]) / (1 + step), 4),
                        }
                    )
                if len(hypothesis) >= self.beam_size:
                    hypothesis = hypothesis[: self.beam_size]
                    break

            cand_mask = ~node_indices.eq(self.node_vocabulary.eos_index)
            cand_final_scores = cand_final_scores.masked_select(mask=cand_mask)[: self.beam_size]
            beam_ids = beam_ids.masked_select(mask=cand_mask)[: self.beam_size]
            node_indices = node_indices.masked_select(mask=cand_mask)[: self.beam_size]
            father_indices = next_father_indices.masked_select(mask=cand_mask)[: self.beam_size]
            edge_indices = next_edges.masked_select(mask=cand_mask)[: self.beam_size]

            target_nodes = target_nodes[beam_ids, :]
            target_edges = target_edges[beam_ids, :, :]
            target_out_fathers = target_out_fathers[beam_ids, :]
            target_out_edges = target_out_edges[beam_ids, :]

            father2children = [father2children[int(beam_id)] for beam_id in beam_ids]

            target_nodes[:, step + 1] = node_indices
            for i in range(target_edges.size(0)):
                target_edges[i, step + 1, father_indices[i]] = \
                    target_edges[i, father_indices[i], step + 1] = edge_indices[i]
                target_edges[i, step + 1, step + 1] = self.edge_vocabulary.self_loop_index  # self loop
            target_out_fathers[:, step] = father_indices
            target_out_edges[:, step] = edge_indices

            for i in range(target_out_fathers.size(0)):
                father = int(father_indices[i])
                if father not in father2children[i]:
                    father2children[i][father] = []
                else:
                    sibling = father2children[i][father][-1]
                    target_edges[i, sibling, step + 1] = \
                        target_edges[i, step + 1, sibling] = self.edge_vocabulary.sibling_index     # sibling
                father2children[i][father].append(step + 1)

            final_scores = cand_final_scores

        hypothesis.sort(key=lambda e: e['final_score'], reverse=True)
        return hypothesis
