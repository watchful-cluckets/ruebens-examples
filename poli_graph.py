import os, math, inspect
from IPython.display import display_html
from operator import mul
import graphlab as gl
from graphreduce.graph_wrapper import GraphWrapper

this_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
cache_dir = this_dir+'/.twitter_politics/'
if os.path.exists(cache_dir+'parent'):
    gw = GraphWrapper.from_previous_reduction(cache_dir)
else:
    v_path = '/home/kcavagnolo/downloads/TheDemocrats_GOP.vertex.csv.gz'
    e_path = '/home/kcavagnolo/downloads/TheDemocrats_GOP.edge.csv.gz'
    gw, mdls = GraphWrapper.reduce(v_path, e_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    gw.save(cache_dir)

def display_table(rows):
    table_template = '<table>%s</table>'
    row_template = '<tr>%s</tr>'
    header_column_template = '<th>%s</th>'
    normal_column_template = '<td>%s</td>'
    rows_html = []
    for i, row in enumerate(rows):
        row_html = []
        for column in row:
            col_template = header_column_template if i == 0 else normal_column_template
            row_html.append(col_template % column)
        rows_html.append(row_template % ''.join(row_html))
    display_html(table_template % ''.join(rows_html), raw=True)

def mk_labels_html(labels):
    labels_html_template = '<span style="color:#0000FF;padding:5px;">%s</span>'
    labels_html = []
    for x in labels:
        labels_html.append(labels_html_template % x)
    return ''.join(labels_html)

def display_communities(results, score_name, header):
    output_rows = [header]
    for x in results:
        output_row = [str(x[score_name])[:4], x['member_count'], mk_labels_html(x['top_labels'])]
        output_rows.append(output_row)
    display_table(output_rows) 

min_members = 25
communities = gw.g.get_vertices()
communities = communities[communities['member_count'] >= min_members]
display_html('<h3>Popular communities</h3>', raw=True)
header = ['Pagerank', 'Member count', 'Top labels']
display_communities(communities.sort('pr', ascending=False)[:10], 'pr', header)

def reciprocal_interest(scores):
    def _score(row):
        return row['user_interest'] * row['community_interest']
    return scores.apply(_score)

user_community_scores = gw.child.user_community_scores(reciprocal_interest, min_members)

def users_top_communities(user_id, scores):
    user_scores = scores[scores['user_id'] == user_id]
    user_scores = user_scores.join(communities, {'community_id':'__id'})
    user_scores.remove_column('community_id.1')
    return user_scores.sort('score', ascending=False)

header = ['Score', 'Member count', 'Top labels']

display_html('<h3>DNC communities</h3>', raw=True)
dem_id = '14377605'
dem_communities = users_top_communities(dem_id, user_community_scores)
display_communities(dem_communities[:10], 'score', header)

display_html('<h3>RNC communities</h3>', raw=True)
rep_id = '11134252'
rep_communities = users_top_communities(rep_id, user_community_scores)
display_communities(rep_communities[:10], 'score', header)

def users_top_users(user_id, scores, feature_ids):
    assert scores['score'].min() >= 0
    scores = scores.groupby('user_id', 
        {'score':gl.aggregate.CONCAT('community_id', 'score')},
        {'num_features':gl.aggregate.COUNT('community_id')})
    scores = scores[scores['num_features'] > len(feature_ids) * .20]
    user_score = scores[scores['user_id'] == user_id][0]
    def distance(row):
        total_distance = 0
        for x in feature_ids:
            score1 = user_score['score'].get(x)
            score2 = row['score'].get(x)
            if score1 and score2:
                dis = abs(score1 - score2)
            elif score1 or score2:
                dis = (score1 or score2) * 2
            else:
                dis = 0
            total_distance+=dis
        return total_distance
    scores['distance'] = scores.apply(distance)
    scores = scores.join(gw.verticy_descriptions, {'user_id':'__id'})
    scores['distance'] = (scores['distance'] - scores['distance'].mean()) \
        / (scores['distance'].std())
    return scores.sort('distance')

feature_ids = list(rep_communities['community_id'][:5])
feature_ids += list(dem_communities['community_id'][:5])
feature_ids = list(set(feature_ids))

def mk_twitter_link(screen_name):
    return '<a target="_blank" href="https://twitter.com/%s">%s</a>' % (screen_name, screen_name)

def display_accounts(results, score_name, header):
    output_rows = [header]
    for x in results:
        output_row = [str(x[score_name])[:4], 
                      mk_twitter_link(x['screen_name']), 
                      x['description']]
        output_rows.append(output_row)
    display_table(output_rows) 

header = ['Distance', 'Account', 'Description']

display_html('<h3>Accounts similar to the DNC</h3>', raw=True)
dem_users = users_top_users(dem_id, user_community_scores, feature_ids)
display_accounts(dem_users[:10], 'distance', header)

display_html('<h3>Accounts similar to the RNC</h3>', raw=True)
rep_users = users_top_users(rep_id, user_community_scores, feature_ids)
display_accounts(rep_users[:10], 'distance', header)

def users_in_between(distances):
    n_dimensions = len(distances)
    _distances = distances[0]
    for x in distances[1:]:
        _distances = _distances.append(x)
    distances = _distances
    distances = distances.groupby('user_id', {'distances':gl.aggregate.CONCAT('distance')})
    def between(row):
        if len(row['distances']) != n_dimensions:
            return None
        x = gl.SArray(row['distances'])
        if x.std() > .15:
            return None
        return x.mean() + x.std()
    distances['distance'] = distances.apply(between)
    distances = distances.dropna().join(gw.verticy_descriptions, {'user_id':'__id'})
    return distances.sort('distance')

display_html('<h3>Of interest to the DNC and RNC</h3>', raw=True)
equidistant_users = users_in_between([dem_users, rep_users])
display_accounts(equidistant_users[:10], 'distance', header)
