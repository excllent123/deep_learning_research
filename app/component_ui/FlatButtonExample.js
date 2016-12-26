import React from '../node_modules/react';
import AppBar from '../node_modules/material-ui/AppBar';
import ReactDOM from '../node_modules/react-dom';

const FlatButtonExampleSimple = () => (
  <div>
            <FlatButton label="Default" />
            <FlatButton label="Primary" primary={true} />
            <FlatButton label="Secondary" secondary={true} />
            <FlatButton label="Disabled" disabled={true} />
          </div>
        );      

ReactDOM.render(
          <FlatButtonExampleSimple />,
          document.getElementById('content')
        );