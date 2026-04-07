import streamlit as st
from lanngraph_backend import chatbot
from langchain_core.messages import SystemMessage, HumanMessage


# with st.chat_message('user'):
#     st.text('hi')


# with st.chat_message('assistant'):
#     st.text('how i can help you?')

# user_input = st.chat_input("type here")



# if user_input:
#     with st.chat_message('user'):
#       st.text(user_input)  


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

#loadin the converstaion history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("type here")
if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)

    config1 = {"configurable": {"thread_id": "1"}}
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=config1) 
    ai_message = response['messages'][-1].content

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})  
    with st.chat_message('assistant'):
        st.text(ai_message)