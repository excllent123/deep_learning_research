import React from '../node_modules/react';
import AppBar from '../node_modules/material-ui/AppBar';
import {Tabs, Tab} from 'material-ui/Tabs';
import ReactDOM from '../node_modules/react-dom';

/**
 * A simple example of `AppBar` with an icon on the right.
 * By default, the left icon is a navigation-menu.
 */

import getMuiTheme from 'material-ui/styles/getMuiTheme'

class Nav extends React.Component {

    static childContextTypes =
    {
        muiTheme: React.PropTypes.object
    }

    getChildContext()
    {
        return {
            muiTheme: getMuiTheme()
        }
    }

  render() {
    return (
      <AppBar title="My App">
        <Tabs>
          <Tab label="Item 1" />
          <Tab label="Item 2" />
          <Tab label="Item 3" />
          <Tab label="Item 4" />
        </Tabs>
      </AppBar>
    )
  }
}
ReactDOM.render(<Nav />, document.getElementById('app')); 