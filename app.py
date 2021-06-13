import streamlit as st
from front_end import bank_marketing
from WineQuality.process_winequality import wine_quality
from BalanceScale.frontend import bs

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.selectbox(
            'select Dataset',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()


# dummy function
def foo():
    st.title("Hello Foo")


def bar():
    st.title("Hello Bar")


if __name__ == '__main__':
    st.title('')
    st.text("Project: Imbalanced")

    # st.sidebar.title('Dataset')
    # option = st.sidebar.selectbox(
    #     'Dataset selection ',
    #     ['Bank_marketing', 'Wine_quality', 'Scale', 'Adult'])
    # st.header('selected Dataset：' + option)
    # if option == "Bank_marketing":
    #     display_dataset()
    # elif option == "Wine_quality":
    #     pass
    # elif option == "Scale":
    #     pass
    # else:
    #     pass
    app = MultiApp()
    app.add_app("Bank Marketing", bank_marketing)
    app.add_app("Wine Quality", wine_quality)
    app.add_app("Balance scale", bs)
    app.add_app("Adult dataset", bar)
    app.run()
