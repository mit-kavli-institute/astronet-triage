import streamlit as st 
import getpass

class AnnotationHandler:

    def __init__(self, df, astro_id):
        default_uid = getpass.getuser()
        annotator_name = st.text_input("Annotator Name", value=default_uid)
        label_1 = st.radio("First label (Planet/Eclipsing/Unknown/Junk)", ['p', 'e', 'j'], key="label_1")
        label_2 = st.radio("Second label (on Target/Background/Unknown)", ['t', 'b', 'u'], key="label_2")
        agree_with_astronet = st.radio("Agree with Astronet?", ['y', 'n'], key="label_3")
        notes = st.text_input("Notes (optional)")

        if label_1 and annotator_name:
            if st.button("Save annotation and move on"):
                annotation = {
                    "annotator": annotator_name,
                    "astro_id": astro_id,
                    "label_1": label_1,
                    "label_2": label_2,
                    "agree_with_astronet": agree_with_astronet,
                    "notes": notes,
                }
                st.write(annotation)
            if st.button("Skip this one"):
                st.session_state.skipped_astro_ids.append(astro_id)
                st.rerun()
        else:
            st.info("Set the label and enter your name to continue.")