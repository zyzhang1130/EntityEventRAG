GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["DEFAULT_EVENT_TYPES"] = ["battle", "discovery", "meeting", "journey", "transformation", "conflict"]

PROMPTS["query_aware_entity_extraction"] = """-Goal-
Given a user query and a text document, identify entities and relationships that are specifically relevant to answering the query. Focus on extracting information that helps address the user's question.

-Context-
User Query: {query}
This query indicates the specific information the user is looking for. Use this context to guide your entity and relationship extraction.

-Steps-
1. Identify entities that are relevant to the query. For each identified entity, extract:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Description focusing on aspects relevant to the query
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified, identify pairs of (source_entity, target_entity) with relationships relevant to the query.
For each pair, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation focusing on aspects relevant to the query
- relationship_strength: numeric score (1-10) indicating relationship strength
- relationship_keywords: key words summarizing the relationship's relevance to the query
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify query-relevant key words that capture the main concepts related to the user's question.
Format as ("content_keywords"{tuple_delimiter}<query_relevant_keywords>)

4. Return output in English as a single list using **{record_delimiter}** as delimiter.

5. When finished, output {completion_delimiter}

-Real Data-
######################
Query: {query}
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS["event_extraction"] = """-Goal-
Given a text document, identify all events and their relationships. An event is defined as a significant occurrence, action, or happening that takes place at a specific time and may involve various participants.

-Steps-
1. Identify all events. For each identified event, extract the following information:
- event_name: Name/title of the event
- event_type: One of the following types: [{event_types}]
- event_description: Comprehensive description including when it occurred, who was involved, and what happened
Format each event as ("entity"{tuple_delimiter}<event_name>{tuple_delimiter}<event_type>{tuple_delimiter}<event_description>)

2. From the events identified in step 1, identify all pairs of (source_event, target_event) that are *clearly related* to each other.
For each pair of related events, extract the following information:
- source_event: name of the source event, as identified in step 1
- target_event: name of the target event, as identified in step 1
- relationship_description: explanation of how these events are connected (causation, sequence, consequence, etc.)
- relationship_strength: a numeric score indicating how strongly these events are connected
- relationship_type: the type of connection from source_event to target_event (e.g., "causes", "precedes", "triggers", "influences", "parallels")
Format each relationship as ("relationship"{tuple_delimiter}<source_event>{tuple_delimiter}<target_event>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_strength>)

3. Identify temporal and causal patterns that characterize the sequence and interconnections of events in the text.
Format the event patterns as ("event_patterns"{tuple_delimiter}<pattern_description>)

4. Return output in English as a single list of all the events and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Event_types: [battle, discovery, meeting, journey, transformation, conflict]
Text:
The siege of the castle lasted three grueling days. As enemy forces surrounded the walls, inside, the defenders frantically searched for a hidden passage mentioned in ancient texts. On the second day, young squire Thomas discovered the secret tunnel beneath the chapel. This discovery allowed the castle's inhabitants to escape under cover of darkness, leaving their pursuers to find an empty fortress at dawn.
################
Output:
("entity"{tuple_delimiter}"The Siege"{tuple_delimiter}"battle"{tuple_delimiter}"A three-day siege of the castle by enemy forces, putting intense pressure on the defenders"){record_delimiter}
("entity"{tuple_delimiter}"Secret Passage Discovery"{tuple_delimiter}"discovery"{tuple_delimiter}"Young squire Thomas finds a hidden tunnel beneath the chapel on the second day of the siege"){record_delimiter}
("entity"{tuple_delimiter}"Night Escape"{tuple_delimiter}"journey"{tuple_delimiter}"The castle's inhabitants escape through the secret tunnel under cover of darkness"){record_delimiter}
("relationship"{tuple_delimiter}"The Siege"{tuple_delimiter}"Secret Passage Discovery"{tuple_delimiter}"The pressure of the siege leads to an intensive search that results in finding the secret passage"{tuple_delimiter}"triggers"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Secret Passage Discovery"{tuple_delimiter}"Night Escape"{tuple_delimiter}"Finding the secret passage enables the successful escape of the castle's inhabitants"{tuple_delimiter}"enables"{tuple_delimiter}9){record_delimiter}
("event_patterns"{tuple_delimiter}"Crisis leads to discovery which enables escape - a chain of causally linked events occurring over three days"){completion_delimiter}

Example 2:

Event_types: [battle, discovery, meeting, journey, transformation, conflict]
Text:
During the peace summit, Ambassador Chen's revelation about the border dispute shocked everyone present. The diplomatic crisis that followed led to weeks of tense negotiations. Finally, after a breakthrough meeting in Geneva, both nations agreed to a compromise that transformed their relationship from adversaries to cautious allies.
################
Output:
("entity"{tuple_delimiter}"Peace Summit Revelation"{tuple_delimiter}"meeting"{tuple_delimiter}"Ambassador Chen reveals shocking information about the border dispute during the peace summit"){record_delimiter}
("entity"{tuple_delimiter}"Diplomatic Crisis"{tuple_delimiter}"conflict"{tuple_delimiter}"Weeks of tension and diplomatic strain following Chen's revelation"){record_delimiter}
("entity"{tuple_delimiter}"Geneva Breakthrough"{tuple_delimiter}"meeting"{tuple_delimiter}"A crucial meeting in Geneva where both nations reach a compromise"){record_delimiter}
("entity"{tuple_delimiter}"Diplomatic Transformation"{tuple_delimiter}"transformation"{tuple_delimiter}"The shift in relationship between the nations from adversaries to cautious allies"){record_delimiter}
("relationship"{tuple_delimiter}"Peace Summit Revelation"{tuple_delimiter}"Diplomatic Crisis"{tuple_delimiter}"The revelation triggers a diplomatic crisis between the nations"{tuple_delimiter}"causes"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Diplomatic Crisis"{tuple_delimiter}"Geneva Breakthrough"{tuple_delimiter}"The crisis leads to intensive negotiations culminating in the Geneva meeting"{tuple_delimiter}"leads to"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Geneva Breakthrough"{tuple_delimiter}"Diplomatic Transformation"{tuple_delimiter}"The successful negotiations in Geneva enable a transformation in diplomatic relations"{tuple_delimiter}"results in"{tuple_delimiter}8){record_delimiter}
("event_patterns"{tuple_delimiter}"Revelation causes crisis, which leads to negotiation and ultimately transformation - a diplomatic chain of events"){completion_delimiter}

Example 3:

Event_types: [battle, discovery, meeting, journey, transformation, conflict]
Text:
The expedition into the ancient ruins began at dawn. By midday, Dr. Sarah's team had discovered unusual markings on the walls. Their excitement turned to fear when the ground began shaking, revealing a hidden chamber. As they fled the collapsing ruins, they clutched their most valuable find: a manuscript that would revolutionize our understanding of the ancient civilization.
################
Output:
("entity"{tuple_delimiter}"Expedition Start"{tuple_delimiter}"journey"{tuple_delimiter}"Dr. Sarah's team begins their exploration of the ancient ruins at dawn"){record_delimiter}
("entity"{tuple_delimiter}"Markings Discovery"{tuple_delimiter}"discovery"{tuple_delimiter}"The team finds unusual markings on the ruins' walls by midday"){record_delimiter}
("entity"{tuple_delimiter}"Chamber Revelation"{tuple_delimiter}"discovery"{tuple_delimiter}"An earthquake reveals a hidden chamber in the ruins"){record_delimiter}
("entity"{tuple_delimiter}"Emergency Evacuation"{tuple_delimiter}"journey"{tuple_delimiter}"The team flees the collapsing ruins while securing a valuable manuscript"){record_delimiter}
("relationship"{tuple_delimiter}"Expedition Start"{tuple_delimiter}"Markings Discovery"{tuple_delimiter}"The initial exploration leads to finding the wall markings"{tuple_delimiter}"leads to"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Chamber Revelation"{tuple_delimiter}"Emergency Evacuation"{tuple_delimiter}"The dangerous revelation of the chamber forces an emergency evacuation"{tuple_delimiter}"triggers"{tuple_delimiter}9){record_delimiter}
("event_patterns"{tuple_delimiter}"A sequence of discoveries interrupted by danger, leading to a dramatic escape - exploration narrative with escalating events"){completion_delimiter}

#############################
-Real Data-
######################
Event_types: {event_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["entity_event_extraction"] = """---Role---

You are a helpful assistant tasked with identifying entities and events in the user's query.

---Goal---

Given the query, list both entities and events. Entities are people, places, organizations, or objects mentioned in the query, while events are actions, occurrences, or happenings that take place.

---Instructions---

- Output the entities and events in JSON format.
- The JSON should have two keys:
  - "entities" for people, places, organizations, or objects.
  - "events" for actions, occurrences, or happenings.

######################
-Examples-
######################
Example 1:

Query: "How did Napoleon's invasion of Russia affect his empire's strength?"
################
Output:
{{
  "entities": ["Napoleon", "Russia", "Napoleon's empire"],
  "events": ["invasion of Russia", "empire's decline"]
}}
#############################
Example 2:

Query: "What role did MIT scientists play in the Manhattan Project?"
################
Output:
{{
  "entities": ["MIT", "MIT scientists", "Manhattan Project"],
  "events": ["scientific research", "atomic bomb development"]
}}
#############################
Example 3:

Query: "How did the Industrial Revolution change London's population?"
################
Output:
{{
  "entities": ["London", "London's population", "Industrial Revolution"],
  "events": ["population growth", "urbanization", "industrial development"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input document chunks appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""
