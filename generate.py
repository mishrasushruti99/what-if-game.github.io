from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel
from pathlib import Path
import json
import time
from tqdm import tqdm
import os

MODEL = "gemini-2.5-flash-lite"
MIN_INTERVAL_S = 5  # proactive throttle — keeps us under free-tier RPM
_last_call_at = 0.0

# Per-run instrumentation. RUN_DIR is set in init_run_dir() after inputs are read.
RUN_DIR = None
LOG_PATH = None
LEDGER_PATH = None
STATE_LEDGER = {}  # knot_label -> {state, goal, events, decisions, is_ending, written_at_depth}

def init_run_dir(story_slug):
    global RUN_DIR, LOG_PATH, LEDGER_PATH
    RUN_DIR = Path("runs") / f"{story_slug}_{time.strftime('%Y%m%d-%H%M%S')}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = RUN_DIR / "calls.jsonl"
    LEDGER_PATH = RUN_DIR / "state_ledger.json"
    print(f"[run] logging to {RUN_DIR}/", flush=True)

def log_event(record):
    if LOG_PATH is None:
        return
    record["t"] = time.time()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

class KeyEvent(BaseModel):
    eventId: int
    event: str

class KeyEvents(BaseModel):
    inciting_incident: KeyEvent
    crisis: KeyEvent
    climax: KeyEvent

class Narration(BaseModel):
    paragraphs: str
    button_text_1: str
    button_text_2: str

_call_count = 0

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _generate(system_prompt, user_prompt, label="call", **extra_config):
    global _last_call_at, _call_count
    backoff = 30
    config_kwargs = dict(system_instruction=system_prompt, seed=42, **extra_config)
    _call_count += 1
    call_id = _call_count
    log(f"  → #{call_id} {label} (user prompt {len(user_prompt)} chars)")
    started_at = time.time()
    for attempt in range(8):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        t0 = time.monotonic()
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            _last_call_at = time.monotonic()
            text = response.text or ""
            duration = time.monotonic() - t0
            log(f"  ← #{call_id} {label} ok in {duration:.1f}s ({len(text)} chars)")
            log_event({
                "type": "llm_call",
                "call_id": call_id,
                "label": label,
                "model": MODEL,
                "started_at": started_at,
                "duration_s": duration,
                "attempts": attempt + 1,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response": text,
                "ok": True,
            })
            return text
        except genai_errors.ClientError as e:
            _last_call_at = time.monotonic()
            if getattr(e, "code", None) != 429 or attempt == 7:
                log(f"  ✗ #{call_id} {label} failed after {time.monotonic() - t0:.1f}s: {e}")
                log_event({
                    "type": "llm_call",
                    "call_id": call_id,
                    "label": label,
                    "model": MODEL,
                    "started_at": started_at,
                    "duration_s": time.monotonic() - t0,
                    "attempts": attempt + 1,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "error": str(e),
                    "ok": False,
                })
                raise
            if "PerDay" in str(e):
                log("  ✗ daily free-tier quota exhausted — enable billing or wait until midnight Pacific. Aborting.")
                log_event({
                    "type": "llm_call",
                    "call_id": call_id,
                    "label": label,
                    "model": MODEL,
                    "started_at": started_at,
                    "error": "daily quota exhausted",
                    "ok": False,
                })
                raise
            log(f"  ⏸ #{call_id} {label} rate-limited, sleeping {backoff}s before retry {attempt + 2}/8...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
    raise RuntimeError("unreachable")

def gen_json(system_prompt, user_prompt, response_schema=None, label="json"):
    extra = {"response_mime_type": "application/json"}
    if response_schema is not None:
        extra["response_schema"] = response_schema
    return _generate(system_prompt, user_prompt, label=label, **extra)

def gen_text(system_prompt, user_prompt, label="text"):
    return _generate(system_prompt, user_prompt, label=label)

def plot2tree(plot, char_name, num_nodes=""):
    """
    Convert a plot to a story branching tree.

    Args:
        plot: the plot of a movie
        num_nodes: the number of story nodes in the output tree

    Returns:
        A story branching tree that summarizes the plot
    """

    JSON_SCHEMA = f"""
{{
    "node_1": {{
      "state": "<initial state of {char_name}>",    /* The initial state of the main character. This should NOT contain any important plot point. */
      "goal": "<goal of {char_name} given the current state>",    /* The goal the main character is attempting to reach in the current state. This should starts with 'To ...' */
      "decision": "<key decision taken by {char_name} that propels the story forward>", /* The key decision taken by {char_name} given the state and goal, starting with '{char_name} decides to ...'. */
      "edgeEvents": [                      /* List of specific events resulting from the key decision and leading up to the state of next node. Each event should be a complete sentence with all involved characters */
        "<repeat key decision taken by {char_name} that propels the story forward, starting with '{char_name} decides to ...'>",
        "<event resulting from the key decision and leading to next state>",
        "<next state of {char_name} resulting from the previous events>"
      ],
      "alternate_decision": "<an alternate decision {char_name} could have made given the same state and goal that would have led to a different storyline, starting with '{char_name} decides to ...'>"

    }},
    "node_2":{{
      "state": "<state of the character resulting from the previous node's edgeEvents>",    /* The current state of the main character, resulted from the previous node's edgeEvents. This should NOT contain any important plot point. */
      "goal": "<goal of the character given the current state>",    /* The goal the main character is attempting to reach in the current state. This should starts with 'To ...' */
      "decision": "<key decision taken by {char_name} that propels the story forward>", /* The key decision taken by {char_name} given the state and goal, starting with '{char_name} decides to ...' */
      "edgeEvents": [                      /* List of specific events resulting from the key decision and leading up to the state of next node. Each event should be a complete sentence with all involved characters */
        "<repeat key decision taken by {char_name} that propels the story forward, starting with '{char_name} decides to ...'>",
        "<event resulting from the key decision and leading to next state>",
        "<next state of {char_name} resulting from the previous events>"
      ],
      "alternate_decision": "<an alternate decision {char_name} could have made given the same state and goal that would have led to a different storyline, starting with '{char_name} decides to ...'>"
    }},

    /* ...continue for all {num_nodes} nodes... */

    "node_n": {{     /* n is the total number of nodes */
      "state": "<state of the character resulting from previous node's edgeEvents>",  /* The current state of the main character, resulted from the previous node's edgeEvents. This should NOT contain any important plot point. */
      "goal": "<final character goal given the current state>",    /* The goal the main character is attempting to reach in the final state. This should starts with 'To ...' */
      "decision": "<key decision taken by {char_name} that propels the story forward>", /* The key decision taken by {char_name} given the state and goal, starting with '{char_name} decides to ...' */
      "edgeEvents": [     /* List of final events resulting from the key decision and leading to the end of the story. Each event should be a complete sentence with all involved characters */
        "<repeat key decision taken by {char_name} leading to end of story, starting with '{char_name} decides to ...'>",
        "<event resulting from the key decision and leading to end of story>",
        "<final state of {char_name} resulting from the previous events>"
      ],
      "alternate_decision": "<an alternate decision {char_name} could have made given the same state and goal that would have led to a different storyline, starting with '{char_name} decides to ...'>"
    }}
}}
"""

    tree = gen_json(
        "# You are a helpful fiction writer assistant.",
        f"{plot}\nSummarize the plot above into a plot tree of {'at most 6' if num_nodes == '' else num_nodes} nodes with each node containing the state and goal of {char_name},\
        and the key decision that propels the story forward. Each edge should contain a list of events \
        that lead {char_name} to the state of next node. Also, Given the same state and goal of {char_name}, imagine an alternate decision that would have led {char_name} to a different storyline.\
        Output in JSON format with schema: {JSON_SCHEMA}. Make sure that all important plot points are included in 'edgeEvents' but not in 'state'",
        label=f"plot2tree(num_nodes={num_nodes})",
    )
    return json.loads(tree)

def get_all_events(storyline):
    event_list = []
    for i in storyline.keys():
        event_list.extend(storyline[i]['edgeEvents'])
    return event_list

def get_key_events(events):
    numbered = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(events))
    key_events = gen_json(
        "Here are some definitions in the context of three-act story structure:\
         The inciting incident is an event that pulls the protagonist out of their normal world and into the main action of the story. It is the turning point between Act One and Act Two.\
         The crisis is the moment when the protagonist faces their greatest challenge or obstacle, leading directly to the climax of the story. It is the turning point between Act Two and Act Three.\
         The climax is the climactic confrontation in which the hero faces a point of no return: they must either prevail or perish. It occurs in Act Three and should have the peak tension of the story.\
         You will be given a numbered list of events from a movie plot. For each of inciting_incident, crisis, and climax, return the integer event number (the leading number on the line you choose) and the event text. Every eventId MUST be a positive integer; never null. The three numbers must be strictly increasing.",
        numbered,
        response_schema=KeyEvents,
        label="get_key_events",
    )
    return json.loads(key_events)

def generate_prompt(all_events, key_events, storyline, branching_node, charname):
    key_event_indices = [i['eventId'] for i in list(key_events.values())]
    key_event_list = [i['event'] for i in list(key_events.values())]

    branching_event = (branching_node - 1) * 3 + 1
    if branching_event <= int(key_event_indices[0]):
        mpp = key_event_list
    elif branching_event <= int(key_event_indices[1]):
        mpp = key_event_list[1:]
    elif branching_event <= int(key_event_indices[2]):
        mpp = key_event_list[2]
    else:
        mpp = 'the rest of the story'
    prompt = gen_text(
        "# You are an expert in prompting large language models. Output ONLY the prompt itself as plain text. No JSON, no markdown code fences, no preamble, no commentary.",
        f"Original storyline:{all_events}\n\
         Write a prompt for a large language model with following requirements:\
         1. Ask to use the original storyline as a reference to write an alternate storyline that branches out at event {branching_event} if {char_name} {storyline[f'node_{branching_node}']['alternate_decision']} instead of {storyline[f'node_{branching_node}']['decision']}.\
         2. Provide 5 thought-provoking concrete guiding questions as potential directions to explore that expand the following:\n\
            a. How would alternate decision change or replace {mpp}?\
            b. How would {char_name} make key decisions that overcome new challenges and propel the story forward?\
         3. Describe what an ideal alternate storyline should look like.\
         4. Ask to output the alternate storyline as a list of {(len(storyline) - branching_node + 1) * 3} events that has {storyline[f'node_{branching_node}']['alternate_decision']} as the first event.",
        label=f"generate_prompt(branching_node={branching_node})",
    )
    return prompt

def write_new_storyline(all_events, prompt):
    JSON_SCHEMA = """
{
    "events": {
        "event number": "an event in the new storyline"
    }
}
"""
    new_storyline = gen_json(
        "# You are a helpful fiction writer assistant.",
        f"Original storyline:\n{all_events}\n\n{prompt}\nOutput in JSON format with schema: {JSON_SCHEMA}.",
        label="write_new_storyline",
    )
    return list(json.loads(new_storyline)["events"].values())

def merge_tree(og_storyline, new_storyline, branching_node):
    merge_tree = {}
    for i in range(1, branching_node + len(new_storyline)):
        if i < branching_node:
            merge_tree[f'node_{i}'] = og_storyline[f'node_{i}']
        elif i == branching_node:
            merged_node = og_storyline[f'node_{branching_node}']
            merged_node['decision'] = new_storyline[f'node_1']['decision']
            merged_node['edgeEvents'] = new_storyline[f'node_1']['edgeEvents']
            merged_node['alternate_decision'] = new_storyline[f'node_1']['alternate_decision']
            merge_tree[f'node_{i}'] = merged_node
        else:
            merge_tree[f'node_{i}'] = new_storyline[f'node_{i-branching_node+1}']
    return merge_tree

def narrate(node, char_name, is_ending=False):
    if not is_ending:
        narration = gen_json(
            f"# You are writing a Choose Your Own Adventure style interactive fiction game in which the player is {char_name}.\
            You will be given a list of events, the resulting state and goal of the character, and two decisions.\
            Do the following:\
                1. Narrate each event in a short paragraph. You should never mention {char_name} but always use the second-person perspective.\
                2. Seamlessly transition to the state and goal of the player.\
                3. Provide two short button-text strings reflecting the two decisions.\
            Separate paragraphs with two newline characters inside the paragraphs field.",
            f"{node}",
            response_schema=Narration,
            label="narrate(branch)",
        )
        return json.loads(narration)
    else:
        paragraphs = gen_text(
            f"# You are writing a Choose Your Own Adventure style interactive fiction game in which the player is {char_name}. "
            f"You will be given a list of events. Narrate each event in a short paragraph using second-person perspective, "
            f"never mentioning {char_name} by name. Seamlessly transition to the ending of the story. "
            f"On a final separate line, print 'THE END'. "
            f"Output ONLY the prose itself — no JSON, no markdown fences, no preamble, no commentary. Separate paragraphs with blank lines.",
            f"{node}",
            label="narrate(ending)",
        )
        return {"paragraphs": paragraphs}

def reorder_tree(tree):
    reordered_tree = []
    prev_events = tree['node_1']['edgeEvents']
    if len(tree) > 1:
        for k in sorted(tree.keys())[1:]:
            reordered_tree.append({
                "events": prev_events,
                "state": tree[k]['state'],
                "goal": tree[k]['goal'],
                "original_decision": tree[k]['decision'],
                "alternate_decision": tree[k]['alternate_decision']
            })
            prev_events = tree[k]['edgeEvents']
    reordered_tree.append({
        "events": prev_events,
        "state": None,
        "goal": None,
        "original_decision": None,
        "alternate_decision": None
    })
    return reordered_tree

def _record_knot(label, node, is_ending, narration):
    # Detect duplicate knot writes — these are the structural bug we identified earlier.
    duplicate = label in STATE_LEDGER
    STATE_LEDGER[label] = {
        "state": node.get("state"),
        "goal": node.get("goal"),
        "events": node.get("events"),
        "original_decision": node.get("original_decision"),
        "alternate_decision": node.get("alternate_decision"),
        "is_ending": is_ending,
        "button_text_1": narration.get("button_text_1"),
        "button_text_2": narration.get("button_text_2"),
        "duplicate_overwrite": duplicate,
    }
    log_event({
        "type": "knot_written",
        "label": label,
        "is_ending": is_ending,
        "duplicate_overwrite": duplicate,
        "paragraph_chars": len(narration.get("paragraphs", "")),
    })

def add_ink_and_chart(reordered_tree, char_name, prefices, ink, chart):
    log_event({
        "type": "knot_batch",
        "prefices": prefices,
        "tree_size": len(reordered_tree),
        "shape_mismatch": len(reordered_tree) != len(prefices),
    })
    nl = '\n'
    if len(reordered_tree) > 1:
        for i, n in enumerate(reordered_tree[:-1]):
            ink.append(f"== {prefices[i]} ==")
            narration = narrate(n, char_name)
            ink.append(f"{narration['paragraphs']}")
            ink.append(f"+ [{narration['button_text_1']}] -> {prefices[i]}L")
            ink.append(f"+ [{narration['button_text_2']}] -> {prefices[i]}R")
            chart.append(f"{prefices[i]}E({nl.join(n['events'])}) --> {prefices[i]}")
            chart.append(f"{prefices[i]}[[S: {n['state']}{nl}G: {n['goal']}]] --> |{narration['button_text_1']}|{prefices[i]}LE")
            chart.append(f"{prefices[i]} --> |{narration['button_text_2']}|{prefices[i]}RE")
            _record_knot(prefices[i], n, is_ending=False, narration=narration)


    ink.append(f"== {prefices[-1]} ==")
    narration = narrate(reordered_tree[-1], char_name, is_ending=True)
    ink.append(f"{narration['paragraphs']}")
    ink.append('-> END')
    chart.append(f"{prefices[-1]}E({nl.join(reordered_tree[-1]['events'])})")
    _record_knot(prefices[-1], reordered_tree[-1], is_ending=True, narration=narration)

def branch(tree, tree_labels, char_name, max_len, ink, chart, pbar):
    depth = len(tree_labels[0])
    log(f"branch() depth={depth}/{max_len} labels={tree_labels}")
    if depth == max_len:
        log(f"branch() depth={depth} reached max_len, returning")
        return
    all_events = get_all_events(tree)
    key_events = get_key_events(all_events)
    prompts = []
    for i in range(depth + 1, max_len + 1):
        prompts.append(generate_prompt(all_events, key_events, tree, i, char_name))
    log(f"branch() depth={depth} got {len(prompts)} prompts, iterating {len(tree_labels)} labels")
    for i, prefix in enumerate(tree_labels):
        if len(prefix) == max_len:
            break
        log(f"branch() depth={depth} label[{i}]={prefix!r}")
        paths = [prefix + 'R']
        while len(paths[-1]) < max_len:
            paths.append(paths[-1] + 'L')

        new_storyline = write_new_storyline(all_events, prompts[i])
        expected_nodes = int(len(new_storyline) / 3)
        log(f"  new_storyline has {len(new_storyline)} events → expecting {expected_nodes} nodes")
        new_tree = plot2tree(new_storyline, char_name, expected_nodes)
        retry = 0
        while len(new_tree) != expected_nodes:
            retry += 1
            log(f"  plot2tree returned {len(new_tree)} nodes, expected {expected_nodes}; retry #{retry}")
            if retry >= 3:
                log(f"  giving up on exact node count after {retry} retries; proceeding with {len(new_tree)} nodes")
                break
            new_tree = plot2tree(new_storyline, char_name, expected_nodes)
        reordered_tree = reorder_tree(new_tree)
        add_ink_and_chart(reordered_tree, char_name, paths, ink, chart)
        pbar.update(1)
        new_tree = merge_tree(tree, new_tree, len(paths[0]))
        branch(new_tree, paths, char_name, max_len, ink, chart, pbar)

def generate(og_plot, char_name, num_nodes):
    og_tree = plot2tree(og_plot, char_name, num_nodes)
    ink = ["-> S", "== S =="]
    nl = '\n'
    reordered_tree = reorder_tree(og_tree)
    tree_labels = ['L'*i for i in range(1, num_nodes + 1)]
    start_node = {
        "events": None,
        "state": og_tree['node_1']['state'],
        "goal": og_tree['node_1']['goal'],
        "original_decision": og_tree['node_1']['decision'],
        "alternate_decision": og_tree['node_1']['alternate_decision']
    }
    narration = narrate(start_node, char_name)
    ink.append(f"{narration['paragraphs']}")
    ink.append(f"+ [{narration['button_text_1']}] -> L")
    ink.append(f"+ [{narration['button_text_2']}] -> R")
    chart = [f"S[[S: {start_node['state']}{nl}G: {start_node['goal']}]] --> |{narration['button_text_1']}|LE"]
    chart.append(f"S[[S: {start_node['state']}{nl}G: {start_node['goal']}]] --> |{narration['button_text_2']}|RE")
    add_ink_and_chart(reordered_tree, char_name, tree_labels, ink, chart)

    pbar = tqdm(total=num_nodes*num_nodes)
    pbar.update(1)
    branch(og_tree, ['L'*i for i in range(num_nodes + 1)], char_name, len(tree_labels), ink, chart, pbar)

    return ink, chart

PRESETS = {
    "harry_potter": {
        "char_name": "Harry Potter",
        "story_name": "Harry Potter and the Philosopher's Stone",
        "plot": (
            "Harry Potter is an orphan living a miserable life with his cruel aunt and uncle, the Dursleys, "
            "and their spoiled son Dudley. He sleeps in a cupboard under the stairs and is treated terribly. "
            "Strange things keep happening around him that he can't explain. "
            "On his eleventh birthday, a giant named Hagrid arrives and reveals the truth: Harry is a wizard, "
            "and his parents weren't killed in a car crash as the Dursleys claimed. They were murdered by an evil "
            "wizard named Voldemort, who tried to kill baby Harry too but mysteriously failed, leaving Harry with a "
            "lightning-bolt scar on his forehead. Harry is famous in the wizarding world as 'the boy who lived.' "
            "Hagrid takes Harry to Diagon Alley to buy school supplies, including his wand from Ollivanders "
            "(a wand with a phoenix feather core, brother to Voldemort's wand). At Gringotts bank, Hagrid also "
            "retrieves a mysterious package on Dumbledore's orders. "
            "Harry boards the Hogwarts Express from Platform 9¾ and meets Ron Weasley, who becomes his best friend, "
            "and Hermione Granger, a bossy but brilliant student. At Hogwarts, the Sorting Hat places all three in "
            "Gryffindor house, despite considering Slytherin for Harry. "
            "Harry begins his magical education, learning subjects like Potions (taught by the cold Professor Snape, "
            "who seems to despise him), Transfiguration, and Defense Against the Dark Arts (taught by the nervous, "
            "stuttering Professor Quirrell). Harry discovers he's a natural at flying and becomes the youngest "
            "Quidditch Seeker in a century. "
            "Around the midpoint, the trio's friendship is cemented when Harry and Ron save Hermione from a troll on "
            "Halloween. They also start to suspect that Snape is trying to steal something hidden in the school—"
            "something guarded by a giant three-headed dog on the forbidden third floor."
        ),
    },
    # Add more presets here, e.g.:
    # "lord_of_the_rings": {"char_name": "Frodo Baggins", "story_name": "...", "plot": "..."},
}
DEFAULT_PRESET = "harry_potter"

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or input('Please enter your Gemini API Key:\n'),
)

_d = PRESETS[DEFAULT_PRESET]
char_name = input(f"Please enter the character name [{_d['char_name']}]:\n").strip() or _d["char_name"]
story_name = input(f"Please enter the story name [{_d['story_name']}]:\n").strip() or _d["story_name"]
og_plot = input("Please enter the original plot (press Enter to use default):\n").strip() or _d["plot"]
num_nodes = 4

story_slug = story_name.lower().replace(' ', '_')
init_run_dir(story_slug)
log_event({"type": "run_start", "char_name": char_name, "story_name": story_name, "num_nodes": num_nodes, "model": MODEL, "plot_chars": len(og_plot)})

print('Generating storylines...')
try:
    ink, chart = generate(og_plot, char_name, num_nodes)
    log_event({"type": "generate_done", "ok": True, "knot_count": len(STATE_LEDGER)})
except Exception as e:
    log_event({"type": "generate_done", "ok": False, "error": str(e), "knot_count": len(STATE_LEDGER)})
    # Always dump the partial ledger so a failed run is still inspectable.
    with open(LEDGER_PATH, "w") as f:
        json.dump(STATE_LEDGER, f, indent=2, default=str)
    print(f"[run] partial state_ledger written to {LEDGER_PATH}")
    raise

os.makedirs(f"stories/{story_slug}", exist_ok=True)
with open(f"stories/{story_slug}/ink.txt", 'w+') as file:
    file.write('\n'.join(ink))
with open(f"stories/{story_slug}/chart.txt", 'w+') as file:
    file.write('\n'.join(chart))

# Dump state ledger to the run dir AND a copy next to the story for convenience.
with open(LEDGER_PATH, "w") as f:
    json.dump(STATE_LEDGER, f, indent=2, default=str)
with open(f"stories/{story_slug}/state_ledger.json", "w") as f:
    json.dump(STATE_LEDGER, f, indent=2, default=str)

print(f"[run] wrote {len(STATE_LEDGER)} knots; calls log: {LOG_PATH}; ledger: {LEDGER_PATH}")
print('finished')
