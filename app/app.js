import React from 'react';
import ReactDOM from 'react-dom';
import AppBar_V2 from './component_ui/app_bar'
// ERROR : import AppBar_V2 from 'component_ui/app_bar'



class MarkdownEditor extends React.Component {
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
        this.rawMarkup = this.rawMarkup.bind(this);
        this.state = {
            value: 'Type some *markdown* here!',
        }
    }
    handleChange() {
        this.setState({value: this.refs.textarea.value});
    }
    // 將使用者輸入的 Markdown 語法 parse 成 HTML 放入 DOM 中，
    // React 通常使用 virtual DOM 作為和 DOM 溝通的中介，不建議直接由操作 DOM。
    // 故使用時的屬性為 dangerouslySetInnerHTML
    rawMarkup() {
        const md = new Remarkable();
        return { __html: md.render(this.state.value) };
    }
    render() {
        return (
          <div className="MarkdownEditor">
            <h3>Input</h3>
            <textarea
              onChange={this.handleChange}
              ref="textarea"
              defaultValue={this.state.value} />
            <h3>Output</h3>
            <div
              className="content"
              dangerouslySetInnerHTML={this.rawMarkup()}
            />
          </div>
        );
    }
}
export default MarkdownEditor;

class Home extends React.Component{
  componentDidMount() {
    const options = { valueNames: [ 'name' ] };
    const userList = new List(this.refs.users, options);
  }

  render() { 
    return (
    /*<AppBar_V2 /> It should be wrapped in a parent element. e.g

 return(
      <div id="parent">
        <div id="div1"></div>
        <div id="div1"></div>
      </div>
      )*/
    <div ref="parent">
    <div ref="users">
      <input className="search" placeholder="Search" />
      <ul className="list">
        <li><h3 className="name">Jonny Stromberg</h3></li>
        <li><h3 className="name">Jonas Arnklint</h3></li>
        <li><h3 className="name">Martina Elm</h3></li>
      </ul>
    </div>
    <div className="MarkdownEditor"></div>
    </div>
    );
  }
};


//ReactDOM.render(<AppBar_V2 />, document.getElementById('app'));

ReactDOM.render(<Home />, document.getElementById('app2'));


