
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

ReactDOM.render(<MarkdownEditor />, document.getElementById('app'));



class MyComponent extends React.Component {
  constructor(props) {
    super(props);
    console.log('constructor');
    this.handleClick = this.handleClick.bind(this);
    this.state = {
      name: 'Mark',
    }
  }
  handleClick() {
    this.setState({'name': 'Zuck'});
    //window.alert('dd')

  }
  componentWillMount() {
    console.log('componentWillMount');
  }
  componentDidMount() {
    console.log('componentDidMount');
  }
  componentWillReceiveProps() {
    console.log('componentWillReceiveProps');
  }
  componentWillUpdate() {
    console.log('componentWillUpdate');
  }
  componentDidUpdate() {
    console.log('componentDidUpdate');
  }
  componentWillUnmount() {
    console.log('componentWillUnmount');
  }
  render() {
    return (
      // HRER, handle the basic router of Virtual-Dom to Actual-Dom
      <div onClick={this.handleClick}>Hi, {this.state.name}</div>
    );
  }
}

ReactDOM.render(<MyComponent />, document.getElementById('app2'));