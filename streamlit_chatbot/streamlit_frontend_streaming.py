import streamlit as st
from lanngraph_backend import chatbot
from langchain_core.messages import SystemMessage, HumanMessage


#what is streaming
#in llms, streaming means the model starts sending tokens(words) as soon as they're generated , instead for the entire response to be ready before returning it

#why streaming
#faster response time low drop-off rates
#mimics human like conversation (builds trust, feels alive and keeps the user engaged)
#important in multi modals
#Better ux for long putput such as code
#you can cancel midway saving tokens
#ypu can interleave ui updates eg show thinking.. show tool results



#python generator is a special type pf iterator that allows you to generate valuse on the fly one at a time, using the yield keyowrd instead of return

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

#loadin the converstaion history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("type here")
if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # first add the message to message_history
    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config= {'configurable': {'thread_id': 'thread-1'}},
                stream_mode= 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})